import "./index.css"
import React from 'react';
import { createRoot } from 'react-dom/client';  // ✅ correct import for React 18
import App from './App';                         // ✅ make sure App.js exists

const container = document.getElementById('root');
const root = createRoot(container);             // ✅ create a root
root.render(<App />);                           // ✅ render App component
