// import React, { useState } from 'react';

// function UploadForm() {
//   const [image, setImage] = useState(null);
//   const [prediction, setPrediction] = useState("");

//   const handleImageChange = (e) => {
//     setImage(e.target.files[0]);
//   };

//   const handleSubmit = async (e) => {
//     e.preventDefault();
//     if (!image) return;

//     const formData = new FormData();
//     formData.append("image", image);

//     try {
//       const response = await fetch("http://localhost:8000/predict", {
//         method: "POST",
//         body: formData,
//       });
//       const data = await response.json();
//       setPrediction(data.prediction || data.error);
//     } catch (err) {
//       setPrediction("Error during prediction");
//     }
//   };

//   return (
//   <div className="p-4">
//     <form onSubmit={handleSubmit}>
//       <input
//         type="file"
//         name="image" // 👈 add this
//         accept="image/*"
//         onChange={handleImageChange}
//         required
//       />
//       <button type="submit" className="ml-2 px-4 py-1 bg-green-500 text-white rounded">
//         Predict
//       </button>
//     </form>
//     {prediction && (
//       <div className="mt-4 font-bold">Prediction: {prediction}</div>
//     )}
//   </div>
// );
// }

// export default UploadForm;

import React, { useState } from 'react';

function UploadForm() {
  const [image, setImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [prediction, setPrediction] = useState("");

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(file);
      setPreviewUrl(URL.createObjectURL(file));
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!image) return;

    const formData = new FormData();
    formData.append("image", image);

    try {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      setPrediction(data.prediction || data.error);
    } catch (err) {
      setPrediction("Error during prediction");
    }
  };

  return (
    <div className="container">
      <h1>Plant Disease Detection</h1>
      <form onSubmit={handleSubmit}>
        <label htmlFor="imageUpload" className="upload-label">Upload Image</label>
        <input
          id="imageUpload"
          type="file"
          accept="image/*"
          onChange={handleImageChange}
          required
        />
        {previewUrl && <img src={previewUrl} alt="Preview" />}
        <button type="submit">Predict</button>
      </form>
      {prediction && <div className="result">Prediction: {prediction}</div>}
    </div>
  );
}

export default UploadForm;
