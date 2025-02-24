const bcrypt = require('bcrypt');
const jwt = require('jsonwebtoken');
const UserModel = require("../Models/User");
require('dotenv').config();
const twilio = require('twilio');
const nodemailer = require('nodemailer');
const accountSid = process.env.TWILIO_ACCOUNT_SID;
const authToken = process.env.TWILIO_AUTH_TOKEN;

const client = twilio(accountSid, authToken);
let otpStorage = {};
const transporter = nodemailer.createTransport({
    service: 'gmail',
    auth: {
        user: process.env.EMAIL_USER,
        pass: process.env.EMAIL_PASS
    }
});

const sendOtpEmail = async (email, otp) => {
    const mailOptions = {
        from: "GEN AI Med 3D",
        to: email,
        subject: 'OTP For Registration',
        text: `Your OTP code is ${otp}`
    };

    await transporter.sendMail(mailOptions);
};

const sendOtp = async (req, res) => {
    try {
        const { phone, email, method } = req.body;

        if (method === 'phone') {
            await client.verify.v2.services(process.env.TWILIO_SERVICE_VID)
                .verifications
                .create({
                    to: `+91${phone}`,
                    channel: 'sms'
                });
            res.status(200).json({ message: 'OTP sent to your phone', success: true });
        } else if (method === 'email') {
            const otp = Math.floor(100000 + Math.random() * 900000).toString();
            otpStorage[email] = otp;
            await sendOtpEmail(email, otp);
            res.status(200).json({ message: 'OTP sent to your email', success: true, otp });
        }
    } catch (err) {
        console.error("Error sending OTP:", err);
        res.status(500).json({ message: "Internal server error", success: false });
    }
};

const verifyOtp = async (req, res) => {
    try {
        const { phone, email, otp, method, name, password } = req.body;
        let isValidOtp = false;
        if (method === 'phone') {
            const verificationCheck = await client.verify.v2.services(process.env.TWILIO_SERVICE_VID)
                .verificationChecks
                .create({ to: `+91${phone}`, code: otp });

            if (verificationCheck.status !== 'approved') {
                return res.status(400).json({ message: "Invalid OTP", success: false });
            }
        } else if (method === 'email') {
            isValidOtp = otp === otpStorage[email];
            delete otpStorage[email];

            if (!isValidOtp) {
                return res.status(400).json({ message: "Invalid OTP", success: false });
            }
        }

        const hashedPassword = await bcrypt.hash(password, 10);
        const newUser = new UserModel({ name, email, phone, password: hashedPassword });
        await newUser.save();

        res.status(201).json({ message: "User registered successfully", success: true });
    } catch (err) {
        console.error('Error during OTP verification:', err);
        res.status(500).json({ message: "Internal server error", success: false });
    }
};
const login = async (req, res) => {
    try {
        const { name, email, password } = req.body;
        const user = await UserModel.findOne({ name });
        const errorMs = 'Auth failed email or password is wrong';
        if (!user) {
            return res.status(403)
                .json({ message: errorMs, success: false });
        }
        const isPassEqual = await bcrypt.compare(password, user.password);
        if (!isPassEqual) {
            return res.status(403)
                .json({ message: errorMs, success: false });
        }
        const jwtToken = jwt.sign(
            { email: user.email, _id: user._id },
            process.env.JWT_SECRET,
            { expiresIn: '24h' })

        res.status(200)
            .json({
                message: "Login success",
                success: true,
                jwtToken,
                email,
                name: user.name
            })
    } catch (err) {
        res.status(500)
            .json({
                message: "Internal server error",
                success: false
            })
    }
}


module.exports = { sendOtp, verifyOtp, sendOtpEmail, login }