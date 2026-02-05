"""
Ingestion pipeline commands for TechDoc Genie.
"""
import click
from pathlib import Path


@click.group()
def ingestion_group():
    """Commands for ingestion pipelines and vector store management."""
    pass


@ingestion_group.command("build-vectorstore")
@click.option(
    "--data-dir",
    "-d",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Directory containing raw documents (HTML/PDF files).",
)
@click.option(
    "--output-dir",
    "-o",
    default="data/vector_store",
    type=click.Path(file_okay=False, dir_okay=True),
    help="Directory to save the vector store. Default: data/vector_store",
)
@click.option(
    "--chunk-size",
    "-c",
    type=int,
    default=512,
    help="Size of text chunks. Default: 512",
)
@click.option(
    "--chunk-overlap",
    type=int,
    default=50,
    help="Overlap between chunks. Default: 50",
)
@click.option(
    "--chunk-strategy",
    "-s",
    type=click.Choice(["recursive", "token", "semantic"]),
    default="recursive",
    help="Chunking strategy to use. Default: recursive",
)
@click.option(
    "--doc-format",
    "-f",
    type=click.Choice(["html", "pdf", "auto"]),
    default="auto",
    help="Document format to load. Default: auto (both HTML and PDF)",
)
@click.option(
    "--no-split-sections",
    is_flag=True,
    help="Disable intelligent PDF section splitting (use pages instead).",
)
@click.option(
    "--name",
    "-n",
    type=str,
    default=None,
    help="Name for the vector store. If not provided, auto-generated from config.",
)
@click.option(
    "--test-query",
    "-t",
    type=str,
    default=None,
    help="Optional test query to run after building the vector store.",
)
def build_vectorstore(
    data_dir: str,
    output_dir: str,
    chunk_size: int,
    chunk_overlap: int,
    chunk_strategy: str,
    doc_format: str,
    no_split_sections: bool,
    name: str,
    test_query: str,
):
    """
    Build a vector store from documentation files.
    
    This command:
    1. Loads documents (HTML/PDF) from the data directory
    2. Chunks them using the specified strategy
    3. Generates embeddings using HuggingFace
    4. Creates and saves a vector store
    
    Examples:
    \b
    # Basic usage
    techdoc-genie ingest build-vectorstore -d data/raw/postgresql
    
    \b
    # Custom chunk size and strategy
    techdoc-genie ingest build-vectorstore -d data/raw/postgresql -c 1024 -s semantic
    
    \b
    # With test query
    techdoc-genie ingest build-vectorstore -d data/raw/postgresql -t "How do I create a table?"
    """
    import sys
    from src.ingestion.document_loader import DocumentLoader
    from src.ingestion.chunker import DocumentChunker
    from src.ingestion.providers.huggingface import HuggingFaceEmbeddingProvider
    from src.retrieval.vector_store import VectorStore
    from src.utils.logger import setup_logger
    
    logger = setup_logger(__name__)
    
    # Display configuration
    click.echo("=" * 80)
    click.echo("Building Vector Store")
    click.echo("=" * 80)
    click.echo(f"Data directory: {data_dir}")
    click.echo(f"Output directory: {output_dir}")
    click.echo(f"Chunk size: {chunk_size}")
    click.echo(f"Chunk overlap: {chunk_overlap}")
    click.echo(f"Chunk strategy: {chunk_strategy}")
    click.echo(f"Document format: {doc_format}")
    click.echo(f"PDF section splitting: {'Disabled' if no_split_sections else 'Enabled'}")
    click.echo("=" * 80)
    
    try:
        # Step 1: Load documents
        click.echo(f"\n📂 Step 1: Loading documents from {data_dir}...")
        loader = DocumentLoader(
            data_dir=data_dir,
            doc_format=doc_format,
            split_pdf_sections=not no_split_sections
        )
        documents = loader.load_and_prepare()
        
        if not documents:
            click.echo(f"❌ No documents found in {data_dir}", err=True)
            click.echo("   Make sure you have .html or .pdf files in the directory", err=True)
            raise click.Abort()
        
        click.echo(f"✅ Loaded {len(documents)} documents")
        
        # Show document breakdown
        html_docs = sum(1 for d in documents if d.metadata.get('file_type') == 'html')
        pdf_docs = sum(1 for d in documents if d.metadata.get('file_type') == 'pdf')
        click.echo(f"   - HTML documents: {html_docs}")
        click.echo(f"   - PDF documents: {pdf_docs}")
        
        # Step 2: Chunk documents
        click.echo(f"\n✂️  Step 2: Chunking documents with '{chunk_strategy}' strategy...")
        chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        if chunk_strategy == "recursive":
            chunks = chunker.chunk_recursive(documents)
        elif chunk_strategy == "token":
            chunks = chunker.chunk_by_tokens(documents)
        elif chunk_strategy == "semantic":
            chunks = chunker.chunk_semantic(documents)
        
        click.echo(f"✅ Created {len(chunks)} chunks")
        click.echo(f"   - Average chunks per document: {len(chunks) / len(documents):.1f}")
        
        # Step 3: Initialize embedder
        click.echo(f"\n🤖 Step 3: Initializing HuggingFace embedder...")
        embedder = HuggingFaceEmbeddingProvider()
        click.echo("✅ Embedder ready")
        
        # Step 4: Create vector store
        if name is None:
            name = f"vectorstore_chunk{chunk_size}_overlap{chunk_overlap}"
        
        vector_store_dir = Path(output_dir) / name
        
        click.echo(f"\n💾 Step 4: Creating vector store...")
        vector_store = VectorStore(embedder=embedder, persist_path=str(vector_store_dir))
        vector_store.create_from_documents(chunks)
        click.echo(f"✅ Vector store created in memory")
        
        # Step 5: Save to disk
        click.echo(f"\n💾 Step 5: Saving vector store...")
        vector_store.save(name=name)
        click.echo(f"✅ Vector store saved to: {vector_store_dir}")
        
        # Optional: Test query
        if test_query:
            click.echo(f"\n🔍 Step 6: Testing with query: '{test_query}'")
            results = vector_store.similarity_search(test_query, k=3)
            
            click.echo("\nTop 3 results:")
            click.echo("-" * 80)
            for i, doc in enumerate(results, 1):
                filename = doc.metadata.get('filename', 'unknown')
                section = doc.metadata.get('section_title', 'N/A')
                preview = doc.page_content[:100].replace('\n', ' ')
                
                click.echo(f"\n#{i} {filename}")
                if section != 'N/A':
                    click.echo(f"   Section: {section}")
                click.echo(f"   Preview: {preview}...")
        
        # Summary
        click.echo("\n" + "=" * 80)
        click.echo("✅ Vector store build completed!")
        click.echo("=" * 80)
        click.echo(f"Documents loaded: {len(documents)}")
        click.echo(f"Chunks created: {len(chunks)}")
        click.echo(f"Vector store location: {vector_store_dir}")
        click.echo(f"\nNext steps:")
        click.echo(f"  • Query the vector store with your RAG system")
        click.echo(f"  • Try different chunk sizes: -c 256, -c 1024, etc.")
        click.echo(f"  • Experiment with strategies: -s token, -s semantic")
        
    except Exception as e:
        click.echo(f"\n❌ Error: {e}", err=True)
        logger.exception("Error building vector store")
        raise click.Abort()


@ingestion_group.command("list-vectorstores")
@click.option(
    "--directory",
    "-d",
    default="data/vector_store",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Directory containing vector stores. Default: data/vector_store",
)
def list_vectorstores(directory: str):
    """
    List available vector stores.
    
    Example:
    \b
    techdoc-genie ingest list-vectorstores
    """
    dir_path = Path(directory)
    
    if not dir_path.exists():
        click.echo(f"❌ Directory does not exist: {directory}", err=True)
        raise click.Abort()
    
    # Find directories that look like vector stores
    vectorstores = [d for d in dir_path.iterdir() if d.is_dir()]
    
    if not vectorstores:
        click.echo(f"No vector stores found in {directory}")
        return
    
    click.echo(f"📊 Vector stores in {directory}:")
    click.echo("=" * 80)
    
    for vs in sorted(vectorstores):
        # Get size
        size = sum(f.stat().st_size for f in vs.rglob('*') if f.is_file())
        size_mb = size / (1024 * 1024)
        
        # Count files
        file_count = len([f for f in vs.rglob('*') if f.is_file()])
        
        click.echo(f"\n📦 {vs.name}")
        click.echo(f"   Size: {size_mb:.2f} MB")
        click.echo(f"   Files: {file_count}")
        click.echo(f"   Path: {vs}")