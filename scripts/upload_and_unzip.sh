#!/bin/bash

echo "Creating temporary VM to unzip files..."
# Create a temporary VM name
VM_NAME="temp-unzip-$(date +%s)"

# Create a startup script that will unzip the files
cat > startup-script.sh << 'EOF'
#!/bin/bash
set -x  # Enable debug output
exec 1> >(tee /tmp/startup.log) 2>&1  # Log all output

echo "Starting process..."

# Install required packages
echo "Installing required packages..."
sudo apt-get update
sudo apt-get install -y unzip

echo "Processing zip files..."
cd /tmp

# Process each batch from 1 to 10
for batch in {1..10}; do
    echo "Processing batch $batch..."
    zip_file="product_images_batch_${batch}.zip"
    
    echo "Downloading $zip_file..."
    gsutil cp "gs://finly-mds/$zip_file" .
    
    echo "Creating temporary directory and unzipping files..."
    mkdir -p "extract_batch_${batch}"
    unzip -j "$zip_file" -d "extract_batch_${batch}/"
    
    # Count total files for progress tracking
    TOTAL_FILES=$(find "extract_batch_${batch}" -type f | wc -l)
    echo "Found $TOTAL_FILES files in batch $batch"
    
    # Use gsutil to copy files with progress
    echo "Copying files to GCS..."
    cd "/tmp/extract_batch_${batch}"
    gsutil -m cp -n -r . gs://finly-mds/images/
    
    # Clean up batch files
    cd /tmp
    rm -rf "extract_batch_${batch}"
    rm "$zip_file"
    
    echo "Completed processing batch $batch"
done

# Upload the log file to GCS for later inspection
gsutil cp /tmp/startup.log gs://finly-mds/upload_logs/${VM_NAME}_log.txt

# Create a completion marker file
echo "Process completed at $(date)" > /tmp/completion_marker.txt
gsutil cp /tmp/completion_marker.txt gs://finly-mds/upload_logs/${VM_NAME}_completed.txt

# Self-destruct the VM
gcloud compute instances delete ${VM_NAME} --zone=us-west1-a --quiet
EOF

# Make the startup script executable
chmod +x startup-script.sh

# Create and start the VM with fixed 200GB disk
if ! gcloud compute instances create $VM_NAME \
    --zone=us-west1-a \
    --machine-type=e2-standard-4 \
    --image-family=debian-11 \
    --image-project=debian-cloud \
    --metadata-from-file=startup-script=startup-script.sh \
    --scopes=storage-full \
    --boot-disk-size=200GB \
    --preemptible
then
    echo "Error: Failed to create VM"
    exit 1
fi

# Clean up local files
rm startup-script.sh

echo "Process started successfully!"
echo "The VM will process the files and self-destruct when complete."
echo "You can check the progress by looking at:"
echo "1. Logs: gs://finly-mds/upload_logs/${VM_NAME}_log.txt"
echo "2. Completion status: gs://finly-mds/upload_logs/${VM_NAME}_completed.txt"
echo "3. Final files will be in: gs://finly-mds/images/"

# Verify the uploads
echo "Verifying uploads..."
echo "Checking files in gs://finly-mds/images/"
gsutil ls -r "gs://finly-mds/images/" | wc -l 