#!/bin/bash

# Check if an argument was given
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <dataset-name>"
    exit 1
fi

# Make sure assets directory exists or create it
mkdir -p assets

# Define the URLs for datasets
wiki_text_2="https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip?ref=blog.salesforceairesearch.com"
wiki_text_103="https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip?ref=blog.salesforceairesearch.com"
# shakespeare (manual download): https://www.kaggle.com/datasets/kingburrito666/shakespeare-plays

# Download and unzip dataset based on the argument
case $1 in
    "wiki-text-2")
        wget "$wiki_text_2" -O assets/wikitext-2-v1.zip
        unzip assets/wikitext-2-v1.zip -d assets/
        ;;
    "wiki-text-103")
        wget "$wiki_text_103" -O assets/wikitext-103-v1.zip
        unzip assets/wikitext-103-v1.zip -d assets/
        ;;
    *)
        echo "Unknown dataset name: $1"
        echo "Available datasets: wiki-text-2, wiki-text-103"
        exit 1
        ;;
esac
