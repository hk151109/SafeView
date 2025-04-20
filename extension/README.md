### Step 1: Clone the Repository

```bash
git clone https://github.com/hk151109/SafeView.git
```
### Step 2: Install Dependencies

Navigate to the project directory and install the necessary dependencies:

```bash
npm install
```

### Step 3: Build the Extension

-   Build the extension by running:
    ```bash
    npm run build
    ```

### Step 4: Generate a Release (Optional)

Generate a release zip file to be uploaded to the browser/store:

```bash
npm run release
```

### Step 5: Load the extension in Chromium browsers

-   Go to `chrome://extensions/`.
-   Enable "Developer mode".
-   Click "Load unpacked" and select the project folder.
