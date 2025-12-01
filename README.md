🛡️ AI-Powered Crowd Density Monitoring Using YOLO and Intelligent Alert System
A Case Study on Public Gatherings
CrowdGuard AI is the practical implementation of this case study—a robust web-based surveillance framework designed to prevent crowd disasters (such as stampedes) before they occur. It leverages computer vision techniques and real-time database logging to monitor high-density zones, detect anomalies, and dispatch role-based alerts to security personnel.

🚀 Key Features
🔐 Secure Role-Based Access


3 Distinct Roles: Admin, Security, and Event Planner


Secure Authentication: MongoDB-backed login/signup with email validation


Modern Glassmorphism UI: Responsive interface with smooth animations


📊 Real-Time Monitoring Dashboard


Live Feed Simulation: Placeholder for future YOLOv8 video stream integration


Dynamic Stats: Real-time density gauges, headcounts, and system health indicators


Interactive Visualization: Smooth transitions and professional UI components


🚨 Intelligent Alert System


Anomaly Detection: Logs events such as High Density, Stoppage, Camera Offline


Persistent Storage: All alerts are saved in MongoDB Atlas for long-term tracking


Simulation Mode: Built-in test scenario generator for demonstration and testing



🛠️ Tech Stack
ComponentTechnologyFrontendHTML5, Tailwind CSS, Vanilla JavaScript, Lucide IconsBackendNode.js, Express.jsDatabaseMongoDB Atlas (Cloud), Mongoose ODMSecurityEnvironment Variables (.env), Input ValidationDesignGlassmorphism, CSS Animations, Canvas Particles

📂 Project Structure
CrowdGuard-AI/
├── backend/                  # Server-side logic
│   ├── node_modules/         # Dependencies (Ignored by Git)
│   ├── .env                  # Secrets (Ignored by Git)
│   ├── server.js             # Main Entry Point (API & DB connection)
│   ├── package.json          # Backend dependencies
│   └── package-lock.json
├── public/                   # Client-side files
│   ├── index.html            # Login/Signup Page
│   ├── dashboard.html        # Main Monitoring Dashboard
│   └── landingpage.html      # Marketing/Intro Page
├── .gitignore                # Git configuration
└── README.md                 # Documentation


⚡ Getting Started
Follow these steps to set up the project locally.
Prerequisites


Node.js installed


A MongoDB Atlas account (free tier)


Installation
1. Clone the Repository
git clone https://github.com/YOUR_USERNAME/CrowdGuard-AI.git
cd backend

2. Install Dependencies
cd backend
npm install

3. Configure Environment Variables
Create a .env file inside the backend folder:
MONGO_URI=mongodb+srv://<username>:<password>@cluster0.example.mongodb.net/crowdGuardDB
PORT=5000

4. Run the Server
node server.js

You should see:
🚀 Server running on http://localhost:5000
✅ Connected to MongoDB Atlas

5. Access the Application
Open your browser and visit:
http://localhost:5000/landingpage.html


📸 Screenshots
(Upload screenshots of your Login Page and Dashboard to your repository and link them here.)

🔮 Future Roadmap


YOLOv8 Integration: Connect Python/Flask model to feed live detection data.


SMS/Email Alerts: Use Twilio or Nodemailer for external alerting.


Historical Reporting: Generate downloadable PDF reports of crowd trends.



🤝 Contributing
Contributions are welcome!
Fork the repository and submit a pull request for enhancements or new features.

If you'd like, I can also format this for GitHub Markdown, add badges, or insert screenshots mockups.
