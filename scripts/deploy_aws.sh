#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Full AWS deployment script.
# Creates EC2 instance, security group, uploads models to S3,
# and deploys the API.
#
# Prerequisites:
#   - aws configure done
#   - Docker built locally (docker compose build api)
#   - Models trained (models/registry/ populated)
#
# Usage: bash scripts/deploy_aws.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
AWS_REGION="us-east-1"
INSTANCE_TYPE="t2.micro"          # Free tier eligible
AMI_ID="ami-0c02fb55956c7d316"    # Amazon Linux 2023 us-east-1 (arm64 use ami-0f9de6e2d2f067fca)
KEY_NAME="stock-forecast-key"
SECURITY_GROUP_NAME="stock-forecast-sg"
S3_BUCKET="stock-forecasting-pipeline"
S3_PREFIX="stock-forecast-mlops"
REPO_URL="https://github.com/maniharshith68/stock-forecast-mlops.git"
YOUR_IP=$(curl -s https://checkip.amazonaws.com)

echo "=================================================="
echo "  Stock Forecast MLOps — AWS Deployment"
echo "=================================================="
echo "  Region:    $AWS_REGION"
echo "  Instance:  $INSTANCE_TYPE"
echo "  S3 Bucket: $S3_BUCKET"
echo "  Your IP:   $YOUR_IP"
echo "=================================================="

# ── Step 1: Upload models to S3 ───────────────────────────────────────────────
echo ""
echo "[1/5] Uploading models and data to S3..."
python3 scripts/upload_to_s3.py \
    --bucket "$S3_BUCKET" \
    --prefix "$S3_PREFIX" \
    --region "$AWS_REGION"

# ── Step 2: Create SSH key pair ───────────────────────────────────────────────
echo ""
echo "[2/5] Setting up SSH key pair..."
KEY_FILE="$HOME/.ssh/${KEY_NAME}.pem"

if aws ec2 describe-key-pairs \
    --key-names "$KEY_NAME" \
    --region "$AWS_REGION" \
    --query "KeyPairs[0].KeyName" \
    --output text 2>/dev/null | grep -q "$KEY_NAME"; then
    echo "  Key pair '$KEY_NAME' already exists"
else
    echo "  Creating key pair '$KEY_NAME'..."
    aws ec2 create-key-pair \
        --key-name "$KEY_NAME" \
        --region "$AWS_REGION" \
        --query "KeyMaterial" \
        --output text > "$KEY_FILE"
    chmod 400 "$KEY_FILE"
    echo "  Key saved to: $KEY_FILE"
fi

# ── Step 3: Create security group ────────────────────────────────────────────
echo ""
echo "[3/5] Setting up security group..."

# Check if security group already exists
SG_ID=$(aws ec2 describe-security-groups \
    --group-names "$SECURITY_GROUP_NAME" \
    --region "$AWS_REGION" \
    --query "SecurityGroups[0].GroupId" \
    --output text 2>/dev/null || echo "none")

if [ "$SG_ID" = "none" ] || [ -z "$SG_ID" ]; then
    echo "  Creating security group..."
    SG_ID=$(aws ec2 create-security-group \
        --group-name "$SECURITY_GROUP_NAME" \
        --description "Stock Forecast MLOps API" \
        --region "$AWS_REGION" \
        --query "GroupId" \
        --output text)

    # SSH from your IP only
    aws ec2 authorize-security-group-ingress \
        --group-id "$SG_ID" \
        --protocol tcp --port 22 \
        --cidr "${YOUR_IP}/32" \
        --region "$AWS_REGION"

    # API port — open to all (for testing)
    aws ec2 authorize-security-group-ingress \
        --group-id "$SG_ID" \
        --protocol tcp --port 8000 \
        --cidr "0.0.0.0/0" \
        --region "$AWS_REGION"

    # MLflow port
    aws ec2 authorize-security-group-ingress \
        --group-id "$SG_ID" \
        --protocol tcp --port 5001 \
        --cidr "0.0.0.0/0" \
        --region "$AWS_REGION"

    echo "  Security group created: $SG_ID"
else
    echo "  Security group already exists: $SG_ID"
fi

# ── Step 4: Launch EC2 instance ───────────────────────────────────────────────
echo ""
echo "[4/5] Launching EC2 instance..."

INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SG_ID" \
    --region "$AWS_REGION" \
    --tag-specifications \
        "ResourceType=instance,Tags=[{Key=Name,Value=stock-forecast-api}]" \
    --iam-instance-profile Name=EC2-S3-Access \
    --query "Instances[0].InstanceId" \
    --output text 2>/dev/null || \
  aws ec2 run-instances \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SG_ID" \
    --region "$AWS_REGION" \
    --tag-specifications \
        "ResourceType=instance,Tags=[{Key=Name,Value=stock-forecast-api}]" \
    --query "Instances[0].InstanceId" \
    --output text)

echo "  Instance launched: $INSTANCE_ID"
echo "  Waiting for instance to be running..."

aws ec2 wait instance-running \
    --instance-ids "$INSTANCE_ID" \
    --region "$AWS_REGION"

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids "$INSTANCE_ID" \
    --region "$AWS_REGION" \
    --query "Reservations[0].Instances[0].PublicIpAddress" \
    --output text)

echo "  Instance running: $PUBLIC_IP"

# ── Step 5: Print SSH instructions ───────────────────────────────────────────
echo ""
echo "[5/5] Deployment instructions:"
echo ""
echo "  Wait ~60 seconds for instance to fully boot, then:"
echo ""
echo "  # SSH into the instance"
echo "  ssh -i $KEY_FILE ec2-user@$PUBLIC_IP"
echo ""
echo "  # Once SSH'd in, run:"
echo "  curl -O https://raw.githubusercontent.com/maniharshith68/stock-forecast-mlops/main/infrastructure/ec2_setup.sh"
echo "  bash ec2_setup.sh $REPO_URL $S3_BUCKET $S3_PREFIX"
echo ""
echo "=================================================="
echo "  Instance ID: $INSTANCE_ID"
echo "  Public IP:   $PUBLIC_IP"
echo "  API will be: http://$PUBLIC_IP:8000"
echo "  Docs will be: http://$PUBLIC_IP:8000/docs"
echo "  SSH key: $KEY_FILE"
echo "=================================================="

# Save deployment info for reference
cat > deployment_info.txt << EOF
Instance ID: $INSTANCE_ID
Public IP:   $PUBLIC_IP
API URL:     http://$PUBLIC_IP:8000
Docs URL:    http://$PUBLIC_IP:8000/docs
SSH:         ssh -i $KEY_FILE ec2-user@$PUBLIC_IP
S3 Bucket:   s3://$S3_BUCKET/$S3_PREFIX/
Deployed:    $(date)
EOF

echo ""
echo "  Deployment info saved to: deployment_info.txt"
