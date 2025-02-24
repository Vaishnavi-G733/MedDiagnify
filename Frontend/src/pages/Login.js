import React, { useState } from 'react';
import { ToastContainer, toast } from "react-toastify";
import { Link, useNavigate } from "react-router-dom";
import 'react-toastify/dist/ReactToastify.css';
import { handleSuccess } from '../utils'; 

const Login = () => {
  const [loginInfo, setLoginInfo] = useState({
    email: '',
    password: '',
  });
  const navigate = useNavigate();

  const handleChange = (e) => {
    const { name, value } = e.target;
    setLoginInfo((prevInfo) => ({
      ...prevInfo,
      [name]: value
    }));
  };

  const handleLogin = async (e) => {
    e.preventDefault();
    console.log('LoginInfo->', loginInfo);

    const { email, password } = loginInfo;

    // Simple client-side validation
    if (!email || !password) {
      toast.error('All fields are required!');
      return; // Avoid further execution
    }

    try {
      console.log('fetch')
      const url = "http://172.16.20.125:4000/auth/login";
      const response = await fetch(url, {
        method: "POST",
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(loginInfo)
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const result = await response.json();
      const { success, message, jwtToken, name, error } = result;

      if (success) {
        handleSuccess(message);
        // toast.success("Login successful!");

        // Clear form
        setLoginInfo({
          email: '',
          password: '',
        });

        localStorage.setItem('token', jwtToken);
        localStorage.setItem('loggedInUser', name);

        // Redirect to home page after short delay
        setTimeout(() => {
          navigate('/home');
        }, 1000);

      } else if (error) {
        const details = error?.details?.[0]?.message;
        toast.error(details || "Login failed. Please try again.");
      } else {
        toast.error(message || "Login failed. Please try again.");
      }

    } catch (err) {
      console.error(err);
      toast.error("Login failed. Please try again.");
    }
  };

  return (
    <div className="signup-container">
      <div className="image">
        <img src="https://www.shutterstock.com/image-vector/healthcare-medical-science-icon-digital-260nw-2267264613.jpg" alt="Health Care" />
      </div>
      <form onSubmit={handleLogin} className="signup-form">
        <h1>Login</h1>
        <div>
          <label htmlFor="username">Username:</label>
          <input
            type="text"
            id="username"
            name="username"
            placeholder="Username"
            required
            // value={loginInfo.username}
            // onChange={handleChange}
          />
        </div>
        <div>
          <label htmlFor="email">Email:</label>
          <input
            type="email"
            id="email"
            name="email"
            placeholder="Email"
            required
            value={loginInfo.email}
            onChange={handleChange}
          />
        </div>
        <div>
          <label htmlFor="password">Password:</label>
          <input
            type="password"
            id="password"
            name="password"
            placeholder="Password"
            required
            value={loginInfo.password}
            onChange={handleChange}
          />
        </div>
        <br/>
        <button type="submit">Login</button>
        <span>
          Don't have an account? <Link to="/Signup"><u>Signup</u></Link>
        </span>
        <a href='/password' >forgot password</a>

        <ToastContainer />
      </form>
    </div>
  );
};

export default Login;
