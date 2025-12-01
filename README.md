🛡️ AI-Powered Crowd Density Monitoring Using YOLO and Intelligent Alert System

A Case Study on Public Gatherings

CrowdGuard AI is the practical implementation of this case study—a robust web-based surveillance framework designed to prevent crowd disasters (like stampedes) before they happen. It leverages computer vision concepts and real-time database logging to monitor high-density areas, detect anomalies, and dispatch role-based alerts to security personnel.

🚀 Key Features

🔐 Secure Role-Based Access

3 Distinct Roles: Admin, Security, and Event Planner.

Secure Authentication: MongoDB-backed login/signup with email validation.

Glassmorphism UI: Modern, responsive interface with animated backgrounds.

📊 Real-Time Monitoring Dashboard

Live Feed Simulation: Placeholder for YOLOv8 video stream integration.

Dynamic Stats: Real-time density gauges, head counts, and system health status.

Interactive UI: Smooth transitions and professional data visualization.

🚨 Intelligent Alert System

Anomaly Detection: Logs events like "High Density," "Stoppage," or "Camera Offline."

Persistent Storage: All alerts are saved to MongoDB Atlas—history is never lost.

Simulation Mode: Built-in tool to generate test scenarios for demonstration purposes.

🛠️ Tech Stack

Component

Technology

Frontend

HTML5, Tailwind CSS, Vanilla JavaScript, Lucide Icons

Backend

Node.js, Express.js

Database

MongoDB Atlas (Cloud), Mongoose ODM

Security

Environment Variables (.env), Input Validation

Design

Glassmorphism, CSS Animations, Canvas Particles

📂 Project Structure

CrowdGuard-AI/
├── backend/ # Server-side logic
│ ├── node_modules/ # Dependencies (Ignored by Git)
│ ├── .env # Secrets (Ignored by Git)
│ ├── server.js # Main Entry Point (API & DB connection)
│ ├── package.json # Backend dependencies
│ └── package-lock.json
├── public/ # Client-side files
│ ├── index.html # Login/Signup Page
│ ├── dashboard.html # Main Monitoring Dashboard
│ └── landingpage.html # Marketing/Intro Page
├── .gitignore # Git configuration
└── README.md # Documentation

⚡ Getting Started

Follow these instructions to set up the project locally.

Prerequisites

Node.js installed on your machine.

A MongoDB Atlas account (free tier).

Installation

Clone the repository

git clone [https://github.com/YOUR_USERNAME/CrowdGuard-AI.git](https://github.com/YOUR_USERNAME/CrowdGuard-AI.git)
cd backend

Install Dependencies
Navigate to the backend folder and install the required packages.

cd backend
npm install

Configure Environment Variables
Create a .env file inside the backend folder and add your MongoDB connection string:

MONGO_URI=mongodb+srv://<username>:<password>@cluster0.example.mongodb.net/crowdGuardDB
PORT=5000

Run the Server

node server.js

You should see: 🚀 Server running on http://localhost:5000 and ✅ Connected to MongoDB Atlas.

Access the Application
Open your browser and visit:
http://localhost:5000/landingpage.html

📸 Screenshots

(You can upload screenshots of your Login Page and Dashboard to your repo and link them here later)

🔮 Future Roadmap

$$

 YOLOv8 Integration: Connect Python/Flask model to feed live detection data.


$$

SMS/Email Alerts: Use Twilio or Nodemailer to send physical alerts.

$$

 Historical Reporting: Export PDF reports of crowd trends.

🤝 Contributing

Contributions are welcome! Please fork the repository and create a pull request for any feature enhancements.


$$
