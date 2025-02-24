import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Link } from 'react-router-dom'
import '../Home.css'


function Home() {

  const [loggedInUser, setLoggedInUser] = useState('');
  const navigate = useNavigate();

  useEffect(() => {
    setLoggedInUser(localStorage.getItem("loggedInUser"));
  }, []);
  
  // useEffect(() => {
  //   const user = localStorage.getItem('loggedInUser');
  //   if (user) {
  //     setLoggedInUser(user);
  //   } else {
  //     navigate('/login'); // Redirect to login if not logged in
  //   }
  // }, [navigate]);

  const handleLogout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('loggedInUser');
    setLoggedInUser(''); // Clear state after logout

    setTimeout(() => {
      navigate('/login');
    }, 1000);
  };

  return (
    <div className='home-container'>
      <h1>Home</h1>
      <p>Welcome {loggedInUser}!</p>
      <h2>Prediction using Deep Learning (UNET)</h2>
      <Link to="/DL">
          <button>UNET</button>
      </Link>
      <h2>Prediction using GenAI (UNETR)</h2>
      <Link to="/GenAI">
          <button>UNETR</button>
      </Link>
      <br/>
      <h2>Classify as Benign or Malignant (VIT)</h2>
      <Link to="/Vit">
          <button>VIT</button>
      </Link>
      <br/>
      <button onClick={handleLogout} className='logout-button'>Logout</button>
      
    </div>
  );
}

export default Home;
