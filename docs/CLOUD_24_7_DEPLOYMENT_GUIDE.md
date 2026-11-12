# 24/7 Permanent Cloud Deployment Guide (Runs with Laptop OFF)

To make your IoT Federated Anomaly Detection dashboard run **24 hours a day, 7 days a week, even when your laptop is turned completely off**, the application must run on a **Cloud Server** (an always-on datacenter machine) instead of your local laptop CPU.

We have packaged the entire project into a lightweight **2.55 MB ready-to-deploy archive**:
📁 `cloud_deployment_bundle.zip`

---

## Method 1: Hugging Face Spaces (100% Free Forever, No Credit Card, 3 Minutes)

Hugging Face provides free 24/7 cloud CPU containers for AI projects.

### Step-by-Step:
1. Go to [**huggingface.co**](https://huggingface.co/) and click **Sign Up** (or Log In).
2. Click on your profile icon (top right) → **New Space**.
3. Fill in:
   - **Space name**: `iot-edgeguard`
   - **License**: `mit`
   - **Space SDK**: Select **Docker** (Blank)
   - **Space hardware**: `Free (2 vCPU, 16GB RAM)`
   - **Visibility**: `Public`
4. Click **Create Space**.
5. Click **Files** tab → **Add file** → **Upload files**.
6. Extract `cloud_deployment_bundle.zip` and drag all files directly into the browser upload box, OR upload the files directly.
7. Click **Commit changes**.
8. Hugging Face will automatically build the `Dockerfile` and start the server.
9. **Done!** You get a permanent link:
   `https://huggingface.co/spaces/<YOUR_USERNAME>/iot-edgeguard`
   **This link will stay online 24/7 forever, even if your laptop is shut down or destroyed.**

---

## Method 2: Render.com (Free 24/7 Cloud Web Service)

1. Create a free account at [**render.com**](https://render.com/).
2. Push this project to a free [GitHub](https://github.com/) repository.
3. In Render Dashboard, click **New +** → **Web Service**.
4. Connect your GitHub repository.
5. Render will automatically detect the `Dockerfile` in the root folder.
6. Choose the **Free Instance Type**.
7. Click **Deploy Web Service**.
8. Render assigns a permanent HTTPS link (e.g. `https://iot-edgeguard.onrender.com`).

---

## Method 3: Keep Your Laptop as a Silent 24/7 Home Server (Without Cloud Accounts)

If you don't want to create any cloud accounts and want your current link to keep working without the laptop screen on:

1. Open **Windows Control Panel** → **Hardware and Sound** → **Power Options**.
2. Click **"Choose what closing the lid does"** in the left menu.
3. Change **When I close the lid** (under Plugged in) to: **"Do nothing"**.
4. Double-click `run_forever.bat` in the project root.
5. Plug your laptop into the wall charger, close the laptop screen, and leave it on your desk.
6. The laptop will continue running silently in the background, consuming very little power, and your Cloudflare link will stay active 24/7.

