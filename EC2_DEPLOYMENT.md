# EC2 Deployment Steps for StudyPal

This guide outlines the steps to deploy the StudyPal application on an AWS EC2 instance.

---

## 1. Open Required Ports
- In your EC2 Security Group, open port **8501** for inbound traffic (used by Streamlit).

## 2. Connect to EC2 via SSH
```bash
ssh -i "ec2-key-pair-2.pem" ubuntu@<ec2-public-dns>
```
Replace `<ec2-public-dns>` with your instance's public DNS.

## 3. Install Dependencies
```bash
sudo apt update
sudo apt install python3-venv
sudo apt install tesseract-ocr
sudo apt install poppler-utils
```

## 4. Clone the Project Repository
```bash
git clone <repo-url>
```
If prompted, use your access token as the password.

## 5. Set Up the Project
```bash
cd <project_directory>
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 6. Configure Environment Variables
Create a `.env` file and paste the required values:
```bash
nano .env
```

## 7. Vectorize Data
```bash
python src/vectorize_script.py
```

## 8. Run the Application
### For Development:
```bash
streamlit run src/main.py
```
### For Production (Background):
```bash
nohup streamlit run src/main.py > streamlit.log 2>&1 &
```

## 9. Monitor Logs
```bash
tail -f streamlit.log
```

## 10. Check Running Process
```bash
lsof -i:8501
```

## 11. Stop the Application
```bash
pkill -f "streamlit run src/main.py"
```

## 12. Access the App
Open in your browser:
```
http://<ec2_public_ip>:8501/
```
Replace `<ec2_public_ip>` with your EC2 instance's public IP address.

---

**Notes:**
- Always run the Streamlit app in `nohup` for background execution on production.
- Ensure your `.env` file is correctly configured for the app to work.
- Make sure the vectorization step completes successfully before running the app.

