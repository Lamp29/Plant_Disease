import "./App.css";
import React from 'react';
import UploadForm from './components/UploadForm';

function App() {
  return (
    <div className="container">
      <h1>Plant Disease Detection</h1>
      <UploadForm />
    </div>
  );
}

export default App;
