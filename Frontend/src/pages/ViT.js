import React, { useState } from 'react';
import axios from 'axios';
import { toast } from 'react-toastify';
import '../Home.css';

function Vit() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [classificationResult, setClassificationResult] = useState(null);

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    setSelectedImage(file);
  };

  const handleImageUpload = async () => {
    if (!selectedImage) {
      alert("Please select an image for classification!");
      return;
    }

    const formData = new FormData();
    formData.append("file", selectedImage);

    try {
      const response = await axios.post("http://172.20.10.4:5000/vit_classify", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });
      toast.success("Image classified successfully!");
      console.log("Classification Result: ", response.data);
      setClassificationResult(response.data.output); 
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
    <div className="vit-container">
      <h1>ViT - Image Classification</h1>
      <input 
        type="file" 
        onChange={handleImageChange} 
        style={{ marginBottom: '10px' }} 
      />
      <button onClick={handleImageUpload} className="predict-button">
        Classify Image with ViT
      </button>
      
      {classificationResult && (
        <div className="classification-result">
          <h3>Classification Result:</h3>
          <p>{classificationResult}</p>  {/* Display the predicted label */}
        </div>
      )}
    </div>
  );
}

export default Vit;
