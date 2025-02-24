import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { ToastContainer } from "react-toastify";
import "../Home.css";
import axios from "axios"; // For handling image upload to the Flask backend
import { toast } from "react-toastify";
import { Link } from 'react-router-dom';


function Home() {
  const [loggedInUser, setLoggedInUser] = useState("");
  const [selectedUNETRImage, setSelectedUNETRImage] = useState(null);
  const [unetrResultImage, setUNETRResultImage] = useState(null);
  const [threed,setThreed] = useState(null)
  const navigate = useNavigate();

  useEffect(() => {
    setLoggedInUser(localStorage.getItem("loggedInUser"));
  }, []);

  const handleLogout = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("loggedInUser");
    setTimeout(() => {
      navigate("/login");
    }, 1000);
  };


  const handleUNETRImageChange = (e) => {
    const file = e.target.files[0];
    setSelectedUNETRImage(file);
  };



const handleUNETRImageUpload = async () => {
  if (!selectedUNETRImage) {
      alert("Please select an image for UNETR!");
      return;
  }

  const formData = new FormData();
  formData.append("file", selectedUNETRImage);

  try {
      const response = await axios.post("http://172.20.10.4:5000/upload_augmented", formData, {
          headers: {
              "Content-Type": "multipart/form-data",
          },
      });
      toast.success("UNETR image uploaded successfully!")
      console.log("UNETR Prediction: ", response.data);
      setUNETRResultImage(response.data.augmented_result_path);
      setThreed(response.data.threed)
      if(response.data.threed){
        window.location.href="http://172.20.10.4:5000/papaya"
    }
  } catch (err) {
      handleError(err);
  }
};

  const handleError = (err) => {
    if (err.response) {
      console.error("Server responded with error: ", err.response.data);
    } else if (err.request) {
      console.error("No response received: ", err.request);
    } else {
      console.error("Error in setting up request: ", err.message);
    }
  };

  return (
    <div className="home-container">
      <div className="header">
        <Link to="/home">
          <button>Home</button>
        </Link>
        <h1>UNETR Model</h1>
        <button className="logout-button" onClick={handleLogout}>
          Logout
        </button>
      </div>
      <div className="content">
        <h2 className="title">Welcome {loggedInUser}</h2>



        {/* UNETR Input Section */}
        <div className="unetr-section">
          <h2>UNETR Image Upload</h2>
          <input 
            type="file" 
            onChange={handleUNETRImageChange} 
            style={{ marginBottom: '10px' }} 
          />
          <button onClick={handleUNETRImageUpload} className="predict-button">
            Predict with UNETR
          </button>
          {!threed && unetrResultImage && (
            <div className="result-image">
              <h3>UNETR Result Image:</h3>
              <img 
                src={unetrResultImage} 
                alt="UNETR Result" 
                style={{ width: "500px", height: "auto", borderRadius: '10px' }} 
              />
            </div>
          )}
        </div>
      </div>
      <ToastContainer />
    </div>
  );
}

export default Home;

