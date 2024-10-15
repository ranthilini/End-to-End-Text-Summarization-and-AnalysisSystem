// src/LandingPage.js
import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Container, Button, Typography } from '@mui/material';
import './LandingPage.css'; // Optional: Create a CSS file for landing page styles
import image from './images/image.jpg'; // Adjust the path if necessary

function LandingPage() {
  const navigate = useNavigate();

  const handleNavigate = () => {
    navigate('/app'); // Navigate to the App component
  };

  return (
    <Container style={{ textAlign: 'center', marginTop: '50px' }}>
      <Typography variant="h2" gutterBottom>
        Text Analysis API
      </Typography>
      <img src={image} alt="Text Analysis" style={{ maxWidth: '100%', height: 'auto', marginBottom: '20px' }} />
      <Typography variant="h5" gutterBottom>
        Analyze your text with our powerful text summarization and sentiment analysis tools.
      </Typography>
      <Button variant="contained" color="primary" onClick={handleNavigate}>
        Go to Analysis
      </Button>
    </Container>
  );
}

export default LandingPage;
