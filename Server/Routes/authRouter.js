// const express=require('express')
// const router=express.Router();
// const {signup}=require('../Controllers/AuthController');
// const {signupValidation}=require('../Middlewares/AuthValidation');
// const {login}=require('../Controllers/AuthController');
// const {loginValidation}=require('../Middlewares/AuthValidation');
// const authController = require('../Controllers/AuthController');
// // router.post('/login',(req,res)=>{
// //     res.send('login success');
// // });
// router.post('/login',loginValidation,login);
// router.post('/signup',signupValidation,signup);
// router.post('/sendotp', sendOTP);
// router.post('/verifyotp', verifyOTP);
// router.post('/sendverificationemail', sendVerificationEmail);


// module.exports=router;

const express = require('express');
const router = express.Router();
const { signup, login, sendOtp, verifyOtp, sendVerificationEmail } = require('../Controllers/AuthController');
const { signupValidation, loginValidation } = require('../Middlewares/AuthValidation');




router.post('/login', loginValidation, login);
router.post('/signup', signupValidation, signup);
router.post('/sendotp', sendOtp);
router.post('/verifyotp', verifyOtp);
router.post('/sendverificationemail', sendVerificationEmail);



module.exports = router;

