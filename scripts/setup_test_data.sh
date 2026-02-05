#!/bin/bash
# Setup test data for CLI tests

set -e

echo "Setting up test data for CLI tests..."
echo "======================================"

# Create test data directory
TEST_DATA_DIR="tests/data/raw"
mkdir -p "$TEST_DATA_DIR"

echo ""
echo "📁 Created directory: $TEST_DATA_DIR"

# Download a small test PDF (PostgreSQL cheat sheet)
echo ""
echo "📥 Downloading test PDF..."

# Check if techdoc-genie command exists
if command -v techdoc-genie &> /dev/null; then
    echo "   Using techdoc-genie CLI..."
    techdoc-genie data download-docs \
        "https://www.postgresqltutorial.com/wp-content/uploads/2018/03/PostgreSQL-Cheat-Sheet.pdf" \
        -o "$TEST_DATA_DIR" \
        -n "test_postgres_cheatsheet.pdf"
else
    echo "   techdoc-genie not found, using curl..."
    curl -o "$TEST_DATA_DIR/test_postgres_cheatsheet.pdf" \
        "https://www.postgresqltutorial.com/wp-content/uploads/2018/03/PostgreSQL-Cheat-Sheet.pdf"
fi

# Create a simple HTML test file
echo ""
echo "📄 Creating test HTML file..."

cat > "$TEST_DATA_DIR/test_documentation.html" << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>Test Documentation</title>
</head>
<body>
    <h1>Sample Technical Documentation</h1>
    
    <h2>Chapter 1: Introduction</h2>
    <p>This is a sample technical documentation file used for testing the TechDoc Genie CLI.</p>
    <p>It contains multiple sections and paragraphs to ensure proper chunking and vector store creation.</p>
    
    <h2>Chapter 2: Installation</h2>
    <p>To install the software, follow these steps:</p>
    <ol>
        <li>Download the package from the official website</li>
        <li>Extract the archive to your preferred location</li>
        <li>Run the installation script</li>
        <li>Configure your environment variables</li>
    </ol>
    
    <h2>Chapter 3: Configuration</h2>
    <p>The system can be configured using a YAML configuration file.</p>
    <pre>
    database:
      host: localhost
      port: 5432
      name: mydb
    </pre>
    
    <h2>Chapter 4: Usage</h2>
    <p>Basic usage examples:</p>
    <ul>
        <li>Create a new project: <code>project create myproject</code></li>
        <li>List all projects: <code>project list</code></li>
        <li>Delete a project: <code>project delete myproject</code></li>
    </ul>
    
    <h2>Chapter 5: Troubleshooting</h2>
    <p>Common issues and their solutions:</p>
    <h3>Connection Errors</h3>
    <p>If you encounter connection errors, check your network settings and firewall configuration.</p>
    
    <h3>Performance Issues</h3>
    <p>For performance issues, try increasing the memory allocation or using a more powerful server.</p>
    
    <h2>Chapter 6: API Reference</h2>
    <p>The API provides the following endpoints:</p>
    <ul>
        <li>GET /api/users - List all users</li>
        <li>POST /api/users - Create a new user</li>
        <li>PUT /api/users/:id - Update a user</li>
        <li>DELETE /api/users/:id - Delete a user</li>
    </ul>
</body>
</html>
EOF

echo "   ✅ Created test_documentation.html"

# List what was created
echo ""
echo "✅ Test data setup complete!"
echo "======================================"
echo ""
echo "Files created in $TEST_DATA_DIR:"
ls -lh "$TEST_DATA_DIR"

echo ""
echo "You can now run CLI tests with:"
echo "  pytest tests/cli/ -v"