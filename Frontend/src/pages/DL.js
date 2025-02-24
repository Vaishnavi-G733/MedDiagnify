import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { ToastContainer } from "react-toastify";
import "../Home.css";
import axios from "axios"; // For handling image upload to the Flask backend
import { toast } from "react-toastify";
import { Link } from "react-router-dom";

function Home() {
  const [loggedInUser, setLoggedInUser] = useState("");
  const [selectedUNETImage, setSelectedUNETImage] = useState(null);
  const [unetResultImage, setUNETResultImage] = useState(null);
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

  const handleUNETImageChange = (e) => {
    const file = e.target.files[0];
    setSelectedUNETImage(file);
  };

  const handleUNETImageUpload = async () => {
    if (!selectedUNETImage) {
        alert("Please select an image for UNET!");
        return;
    }

    const formData = new FormData();
    formData.append("file", selectedUNETImage);

    try {
        const response = await axios.post("http://172.20.10.4:5000/upload", formData, {
            headers: {
                "Content-Type": "multipart/form-data",
            },
        });
        toast.success("UNET image uploaded successfully!")
        console.log("UNET Prediction: ", response.data);
        setUNETResultImage(response.data.output_path);
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
        <h1>Predict using Deep Learning Model</h1>

        <button className="logout-button" onClick={handleLogout}>
          Logout
        </button>
      </div>
      <div className="content">
        <h2 className="title">Welcome {loggedInUser}</h2>

        {/* UNET Input Section */}
        <div className="unet-section">
          <h2>UNET Image Upload</h2>
          <input 
            type="file" 
            onChange={handleUNETImageChange} 
            style={{ marginBottom: '10px' }} 
          />
          <button onClick={handleUNETImageUpload} className="predict-button">
            Predict with UNET
          </button>
          {unetResultImage && (
            <div className="result-image">
              <h3>UNET Result Image:</h3>
              <img 
                src={unetResultImage} 
                alt="UNET Result" 
                style={{ width: "500px", height: "auto", borderRadius: '10px' }} 
              />
            </div>
          )}
        </div>

        {/* UNETR Input Section */}
  
      </div>
      <ToastContainer />
    </div>
  );
}

export default Home;

