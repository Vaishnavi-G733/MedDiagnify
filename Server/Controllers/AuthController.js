const bcrypt=require('bcrypt');
const UserModel=require("../Models/User.js");
const jwt=require('jsonwebtoken');
const nodemailer = require('nodemailer');
const { v4: uuidv4 } = require('uuid');
const Otpverification = require('../Models/Otpverification');
const User = require('../Models/User');
// const UserVerification = require('../Models/Userverification.js'); 
const transporter = require('../config/emailConfig.js');
const { generateOtp } = require('../utils/otpUtils.js');
const expiresIn = 300; // or whatever value is appropriate
const twilio = require('twilio');
require('dotenv').config();
const accountSid = process.env.TWILIO_ACCOUNT_SID;
const authToken = process.env.TWILIO_AUTH_TOKEN;

const client = twilio(accountSid, authToken);

async function createVerification() {
    const verification = await client.verify.v2
      .services(process.env.TWILIO_SERVICE_VID)
      .verifications.create({
        channel: "sms",
        to: "+917981853634",
      });
  
    console.log(verification.sid);
  }
  
createVerification();


const signup=async(req,res)=>{
    try{
        
        const{name,email,password}=req.body;
        const user=await UserModel.findOne({email});
        // if(user)
        // {
        //     console.log(user);
        //     return res.status(409)
        //         .json({message:'User is already exist',success:false});
            

        // }
        const userModel=new UserModel({name,email,password});
        userModel.password=await bcrypt.hash(password,10);
        await userModel.save();
        res.status(201)
            .json({
                message:"Signup successfully",
                success:true

            })
    } catch(err){
        res.status(201)
        .json({
            message:"Internal server error",
            success:false

        })
    }
}
const login = async (req, res) => {
    try {
        const { email, password } = req.body;

        // Find user by either email or username
        const user = await UserModel.findOne({ email });
        const errorMsg = 'Authentication failed: incorrect username, email, or password';

        // Check if the user exists
        if (!user) {
            return res.status(401).json({ message: errorMsg, success: false });
        }

        // Compare password
        const isPassEqual = await bcrypt.compare(password, user.password);
        if (!isPassEqual) {
            return res.status(401).json({ message: errorMsg, success: false });
        }

        // Generate JWT token
        const jwtToken = jwt.sign(
            { email: user.email, _id: user._id },
            process.env.JWT_SECRET,
            { expiresIn: '24h' }
        );

        // Successful login response
        return res.status(200).json({
            message: "Login successful",
            success: true,
            jwtToken,
            email,
            name: user.name,
        });

    } catch (err) {
        // Handle internal server error
        return res.status(500).json({
            message: "Internal server error",
            success: false
        });
    }
};



// Send OTP
const sendOtp = async (req, res) => {
    try {
        const { email, phone, method } = req.body;
        
        if (!email && !phone) {
            return res.status(400).json({ success: false, message: 'Email or phone number is required!' });
        }
        
        const otp = generateOtp(); // Generate OTP
        const expiresat = Date.now() + 10 * 60 * 1000; // OTP valid for 10 minutes
        
        const field = email ? { email } : { phone };
        let otpRecord = await Otpverification.findOne(field);
        
        if (otpRecord) {
            otpRecord.otp = otp;
            otpRecord.expiresat = expiresat;
            await otpRecord.save();
        } else {
            await Otpverification.create({ ...field, otp, expiresat });
        }
        
        // Send OTP via email or phone
        if (method == 'email') {
            const otpSent = await sendOtpToUser(email, otp); // Ensure this function is defined
            if (otpSent) {
                return res.json({ success: true, message: 'OTP sent to email successfully!' });
            } else {
                throw new Error('Failed to send OTP to email');
            }
        }

        else if (method == 'phone') {
            const otpSent = await client.verify.v2.services(process.env.TWILIO_SERVICE_VID)
            .verifications
            .create({
                to: `+91${phone}`,
                channel: 'sms'
            });
            console.log(otpSent)
            if (otpSent) {
                return res.json({ success: true, message: 'OTP sent to phone successfully!' });
            } else {
                throw new Error('Failed to send OTP to phone');
            }
        }

    } catch (error) {
        console.error(error);
        res.status(500).json({ success: false, message: 'Failed to send OTP. Please try again.' });
    }
};


// Define the function to send OTP
const sendOtpToUser = async (email, otp) => {
    // console.log(process.env.SMTP_USER,process.env.SMTP_PASS)
    console.log(email,otp)
    const mailOptions = {
        from: process.env.SMTP_USER,
        to: email,
        subject: 'Your OTP Code',
        text: `Your OTP code is ${otp}. It is valid for 10 minutes.`,
    };

    try {
        await transporter.sendMail(mailOptions);
        return true;
    } catch (error) {
        console.error('Error sending OTP:', error);
        return false;
    }
};

// Verify OTP
const verifyOtp = async (req, res) => {
    expirest: new Date(Date.now() + expiresIn * 1000)
    try {
        const { phone,email, otp } = req.body;

        // Validate input
        if (!email || !otp) {
            return res.status(400).json({ success: false, message: 'Email and OTP are required!' });
        }

        // Check if OTP record exists for the email
        const otpverificationrecord = await Otpverification.findOne({ email });

        if (!otpverificationrecord) {
            return res.status(400).json({
                success: false,
                message: "Account has already been verified or doesn't exist. Please sign up again or login!"
            });
        }

        const { expiresat } = otpverificationrecord.expiresat;
        const { otpdb } = otpverificationrecord.otp;

        // Check if OTP has expired
        if (expiresat < Date.now()) {
            await Otpverification.deleteMany({ email });
            return res.status(400).json({
                success: false,
                message: "The code has expired, please request again!"
            });
        }

        // Check if OTP is correct
        if (otp !== otpdb) {
            return res.status(400).json({
                success: false,
                message: "Invalid OTP, please try again (check inbox)"
            });
        }

        // Update user verification status and delete OTP record
        await User.updateOne({ email }, { verified: true });
        await Otpverification.deleteMany({ email });

        res.json({
            success: true,
            message: "Your account has been verified successfully!"
        });

    } catch (error) {
        console.error(error);
        res.status(500).json({
            success: false,
            message: 'Internal server error'
        });
    }
};

// Send verification email
const sendVerificationEmail = ({ _id, email }, res) => {
    const currentUrl = "http://localhost:5000/";
    const uniqueString = uuidv4() + _id;

    const mailOptions = {
        from: process.env.AUTH_EMAIL,
        to: email,
        subject: "Verify your email",
        html: `<p>Verify your email address to complete the signup and login into your account</p>
        <p>This link <b>expires in 6 hours</b>.</p>
        <p>Click <a href=${currentUrl + "user/verify/" + _id + "/" + uniqueString}>here</a> to proceed.</p>`,
    };

    const saltRounds = 10;
    bcrypt
        .hash(uniqueString, saltRounds)
        .then((hashedUniqueString) => {
            const newVerification = new UserVerification({
                userID: _id,
                uniqueString: hashedUniqueString,
                createdAT: Date.now(),
                expiredAT: Date.now() + 21600000, // 6 hours
            });

            newVerification
                .save()
                .then(() => {
                    transporter
                        .sendMail(mailOptions)
                        .then(() => {
                            res.json({
                                status: "SUCCESS",
                                message: "Verification email sent",
                            });
                        })
                        .catch((error) => {
                            console.log(error);
                            res.json({
                                status: "FAILED",
                                message: "Verification email failed",
                            });
                        });
                })
                .catch((error) => {
                    console.log(error);
                    res.json({
                        status: "FAILED",
                        message: "An error occurred while saving verification data",
                    });
                });
        })
        .catch((error) => {
            console.log(error);
            res.json({
                status: "FAILED",
                message: "An error occurred while hashing email data",
            });
        });
};

module.exports = {
    signup,
    login,
    sendOtp,
    sendOtpToUser,
    verifyOtp,
    sendVerificationEmail,
};