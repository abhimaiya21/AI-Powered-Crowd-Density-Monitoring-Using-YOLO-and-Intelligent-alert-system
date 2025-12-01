// 1. Load Environment Variables at the VERY TOP
require('dotenv').config();

const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const bodyParser = require('body-parser');
const path = require('path');

const app = express();
// 2. Use the PORT from .env, or default to 5000
const PORT = process.env.PORT || 5000;

// Middleware
app.use(cors());
app.use(bodyParser.json());

// Serving Static Files (HTML/CSS/JS)
app.use(express.static(path.join(__dirname, '../public')));

// --- MONGODB CONNECTION ---
const MONGO_URI = process.env.MONGO_URI;

if (!MONGO_URI) {
    console.error("❌ FATAL ERROR: MONGO_URI is not defined in .env file");
    process.exit(1);
}

mongoose.connect(MONGO_URI)
.then(() => console.log('✅ Connected to MongoDB Atlas'))
.catch(err => console.error('❌ MongoDB Connection Error:', err));

// --- USER SCHEMA ---
const userSchema = new mongoose.Schema({
    fullName: String,
    email: { type: String, required: true, unique: true },
    password: { type: String, required: true },
    role: { type: String, required: true }
});

const User = mongoose.model('User', userSchema);

// --- HELPER FUNCTION: EMAIL VALIDATION ---
function validateEmail(email) {
    // 1. Basic format check (must have @ and .)
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
        return { isValid: false, message: "Invalid email format." };
    }

    // 2. Specific check for "gmail.co" typo
    if (email.toLowerCase().endsWith('@gmail.co')) {
        return { isValid: false, message: "Invalid domain. Did you mean @gmail.com?" };
    }

    return { isValid: true };
}

// --- ROUTES ---

// 1. Login Route
app.post('/api/login', async (req, res) => {
    const { email, password, role } = req.body;

    try {
        const user = await User.findOne({ email });

        if (!user) {
            return res.status(404).json({ success: false, message: "User not found." });
        }

        if (user.password !== password) {
            return res.status(401).json({ success: false, message: "Invalid password." });
        }

        if (user.role !== role) {
            return res.status(403).json({ success: false, message: `Access denied for role: ${role}` });
        }

        res.json({ success: true, message: `Welcome back, ${user.fullName}!`, user: { name: user.fullName, role: user.role } });

    } catch (err) {
        console.error("Login Error:", err);
        res.status(500).json({ success: false, message: "Server error." });
    }
});

// 2. Signup Route
app.post('/api/signup', async (req, res) => {
    const { fullName, email, password, role } = req.body;

    // --- VALIDATION STEP ---
    const emailValidation = validateEmail(email);
    if (!emailValidation.isValid) {
        return res.status(400).json({ success: false, message: emailValidation.message });
    }

    try {
        // Check if user exists
        const existingUser = await User.findOne({ email });
        if (existingUser) {
            return res.status(400).json({ success: false, message: "Email already registered." });
        }

        // Create new user
        const newUser = new User({ fullName, email, password, role });
        await newUser.save();

        res.status(201).json({ success: true, message: "Account created successfully!" });

    } catch (err) {
        console.error("Signup Error:", err);
        res.status(500).json({ success: false, message: "Error creating account." });
    }
});

// Start Server
app.listen(PORT, () => {
    console.log(`🚀 Server running on http://localhost:${PORT}`);
});