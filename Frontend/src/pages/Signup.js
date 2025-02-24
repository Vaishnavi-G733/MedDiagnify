import React, { useState } from 'react';
import { ToastContainer, toast } from "react-toastify";
import { Link, useNavigate } from "react-router-dom";
import 'react-toastify/dist/ReactToastify.css';
import { handleSuccess } from '../utils'; // Ensure this function does not trigger toast itself
import '../index.css';

const SignUpForm = () => {
  const [signupInfo, setSignupInfo] = useState({
    name: '',
    email: '',
    password: '',
    dob: '',
    phoneotp: '',
    emailotp:'',
    phone: '' // Phone number field
  });
  
  const [otpSent, setOtpSent] = useState({ email: false, phone: false }); // Track OTP sent status
  const navigate = useNavigate();

  const handleChange = (e) => {
    const { name, value } = e.target;
    setSignupInfo((prevInfo) => ({
      ...prevInfo,
      [name]: value
    }));
  };

  const handleSendOtp = async (method) => {
    const { email, phone } = signupInfo;

    // Validate required fields
    if (method === 'email' && !email) {
      toast.error('Email is required to send OTP!');
      return;
    }

    if (method === 'phone' && !phone) {
      toast.error('Phone number is required to send OTP!');
      return;
    }

    try {
      const url = `http://172.16.20.125:4000/auth/sendotp`;
      const response = await fetch(url, 
      {
        method: "POST",
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({email,phone,method})
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const result = await response.json();
      const { status, message } = result;

      if (status === 'PENDING') {
        toast.success(message);
        setOtpSent((prev) => ({ ...prev, [method]: true }));
      } else {
        toast.success(message);
      }

    } catch (err) {
      console.error(err);
      toast.error(`Failed to send OTP via ${method}. Please try again.`);
    }
  };

  const handleSignup = async (e) => {
    e.preventDefault();

    const { name, email, password, dob, phoneotp, emailotp, phone} = signupInfo;

    // Simple client-side validation
    if (!name || !email || !password || !dob || !phoneotp ||!emailotp || !phone) {
      toast.error('All fields are required!');
      return;
    }

    try {
      const url = "http://172.16.20.125:4000/auth/signup";
      const response = await fetch(url, {
        method: "POST",
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(signupInfo)
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const result = await response.json();
      const { success, message, error } = result;

      if (success) {
        handleSuccess(message);
        setSignupInfo({
          name: '',
          email: '',
          password: '',
          dob: '',
          phoneotp:'',
          emailotp: '',
          phone: ''
        });
        setTimeout(() => {
          navigate('/login');
        }, 1000);

      } else if (error) {
        const details = error?.details?.[0]?.message;
        toast.error(details || "Signup failed. Please try again.");
      } else {
        toast.error(message || "Signup failed. Please try again.");
      }

    } catch (err) {
      console.error(err);
      toast.error("Signup failed. Please try again.");
    }
  };

  return (
    <div className="signup-container">
      <div className="image">
        <img src="https://www.shutterstock.com/image-vector/healthcare-medical-science-icon-digital-260nw-2267264613.jpg" alt="Health Care" />
      </div>
      <form onSubmit={handleSignup} className="signup-form">
        <h1>Signup</h1>
        <div className="signup__form__div">
          <label htmlFor="name">Username:</label>
          <input
            onChange={handleChange}
            type="text"
            id="name"
            name="name"
            placeholder="Enter your name"
            required
            value={signupInfo.name}
          />
        </div>
        <div className="signup__form__div">
          <label htmlFor="email">Email:</label>
          <input
            onChange={handleChange}
            type="email"
            id="email"
            name="email"
            placeholder="Email"
            required
            value={signupInfo.email}
          />
        </div>
        <div className="signup__form__div">
          <label htmlFor="phone">Phone Number:</label>
          <input
            onChange={handleChange}
            type="text"
            id="phone"
            name="phone"
            placeholder="Phone number"
            required
            value={signupInfo.phone}
            
          />
        </div>
        <div className="signup__form__div">
          <label htmlFor="password">Password:</label>
          <input
            onChange={handleChange}
            type="password"
            id="password"
            name="password"
            placeholder="Password"
            required
            value={signupInfo.password}
          />
        </div>
        <div className="signup__form__div">
          <label htmlFor="dob">Date of birth:</label>
          <input
            onChange={handleChange}
            type="date"
            id="dob"
            name="dob"
            required
            value={signupInfo.dob}
          />
        </div>
        <div className="signup__form__div">
          <label htmlFor="phoneotp">Phone OTP:</label>
          <input
            onChange={handleChange}
            type="text"
            id="phoneotp"
            name="phoneotp"
            placeholder="Enter Phone OTP"
            // required
            value={signupInfo.phoneotp}
            disabled={otpSent.phone}
          />
        </div>
        <div className="signup__form__div">
          <label htmlFor="emailotp">Email OTP:</label>
          <input
            onChange={handleChange}
            type="text"
            id="emailotp"
            name="emailotp"
            placeholder="Enter Email OTP"
            required
            value={signupInfo.emailotp}
            disabled={otpSent.email} // Only allow input if OTP sent
          />
        </div>
        

        <div className="otp-buttons">
          <button type="button" onClick={() => handleSendOtp('email')} disabled={otpSent.email}>
            {otpSent.email ? 'OTP Sent via Email' : 'Send OTP via Email'}
          </button>
          <button type="button" onClick={() => handleSendOtp('phone')} disabled={otpSent.phone}>
            {otpSent.phone ? 'OTP Sent via Phone' : 'Send OTP via Phone'}
          </button>
        </div>

        <button type="submit">Signup</button>
        <span>
          Already have an account? <Link to="/login"><u>Login</u></Link>
        </span>
        <ToastContainer />
      </form>
    </div>
  );
};

export default SignUpForm;
