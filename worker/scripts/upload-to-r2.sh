#!/bin/bash
# Upload embeddings to R2
#
# Prerequisites:
# 1. Enable R2 in Cloudflare Dashboard
# 2. Run generate-embeddings.ts first to create the files
#
# Usage: ./scripts/upload-to-r2.sh

set -e

BUCKET_NAME="semantic-trail-embeddings"
DATA_DIR="$(dirname "$0")/../data"

echo "Uploading embeddings to R2 bucket: $BUCKET_NAME"

# Create bucket if it doesn't exist
echo "Creating bucket (if not exists)..."
wrangler r2 bucket create "$BUCKET_NAME" 2>/dev/null || echo "Bucket may already exist"

# Upload binary embeddings
echo "Uploading embeddings.bin..."
wrangler r2 object put "$BUCKET_NAME/embeddings.bin" --file="$DATA_DIR/embeddings.bin"

# Upload word index
echo "Uploading embeddings-index.json..."
wrangler r2 object put "$BUCKET_NAME/embeddings-index.json" --file="$DATA_DIR/embeddings-index.json"

echo ""
echo "Upload complete!"
echo "Files uploaded to R2 bucket: $BUCKET_NAME"
echo ""
echo "Next steps:"
echo "1. Uncomment R2 binding in wrangler.toml"
echo "2. Update src/index.ts to load from R2"
echo "3. Run: npm run deploy"
