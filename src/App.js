import React, { useState } from 'react';
import { Button, TextField, Typography, Box, Container, Paper, InputAdornment } from '@mui/material';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom'; // Update this line
import LandingPage from './LandingPage';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [textInput, setTextInput] = useState("");
  const [summary, setSummary] = useState("");
  const [sentiment, setSentiment] = useState("");
  const [keywords, setKeywords] = useState("");
  const [topics, setTopics] = useState("");
  const [searchQuery, setSearchQuery] = useState("");

  const handleFileUpload = (event) => {
    const uploadedFile = event.target.files[0];
    setFile(uploadedFile);
  };

  const handleTextInputChange = (event) => {
    setTextInput(event.target.value);
  };

  const handleSubmit = async () => {
    if (!file && !textInput) {
      alert("Please upload a text file or enter some text.");
      return;
    }

    const formData = new FormData();
    if (file) {
      formData.append('file', file);
    } else {
      formData.append('textInput', textInput);
    }

    try {
      const response = await fetch('http://127.0.0.1:5000/analyze', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorMessage = await response.text();
        alert("Error: " + errorMessage);
        return;
      }

      const result = await response.json();
      setSummary(result.summary);

      const sentimentLabel = `${result.sentiment.label} (${(result.sentiment.score * 100).toFixed(2)}%)`;
      setSentiment(sentimentLabel);

      setKeywords(result.keywords.join(", "));
      setTopics(result.topics.join(" | "));
    } catch (error) {
      alert("An error occurred: " + error.message);
    }
  };

  const handleSearch = (event) => {
    setSearchQuery(event.target.value.toLowerCase());
  };

  const highlightSearchTerm = (content) => {
    if (!searchQuery) return content;
    const regex = new RegExp(`(${searchQuery})`, 'gi');
    const parts = content.split(regex);
    return parts.map((part, index) =>
      regex.test(part) ? <span key={index} className="highlight">{part}</span> : part
    );
  };

  return (
    <Router>
      <Routes> {/* Change Switch to Routes */}
        <Route path="/app" element={(
          <Container maxWidth="md" className="container" style={{ marginTop: '30px' }}>
            <Paper elevation={3} style={{ padding: '30px' }}>
              <Typography variant="h3" className="header" align="center" gutterBottom>
                Text Summarization and Analysis
              </Typography>

              {/* File Upload Section */}
              <div className="file-upload" style={{ marginBottom: '20px' }}>
                <Typography variant="h5">Upload a text file:</Typography>
                <TextField
                  type="file"
                  fullWidth
                  onChange={handleFileUpload}
                  className="upload-button"
                  style={{ marginBottom: '10px' }}
                />
              </div>

              {/* Text Input Section */}
              <div className="text-input" style={{ marginBottom: '20px' }}>
                <Typography variant="h5">Or Enter Text:</Typography>
                <TextField
                  label="Enter text for analysis"
                  multiline
                  rows={4}
                  fullWidth
                  value={textInput}
                  onChange={handleTextInputChange}
                  className="text-input-field"
                  style={{ marginBottom: '10px' }}
                />
              </div>

              <Button
                variant="contained"
                color="primary"
                onClick={handleSubmit}
                className="upload-button"
                fullWidth
                style={{ marginBottom: '20px' }}
              >
                Analyze Text
              </Button>

              {/* Search Bar */}
              <div className="search-bar" style={{ marginBottom: '20px' }}>
                <TextField
                  label="Search Summary, Keywords, or Topics"
                  fullWidth
                  value={searchQuery}
                  onChange={handleSearch}
                  InputProps={{
                    startAdornment: (
                      <InputAdornment position="start">
                        🔍
                      </InputAdornment>
                    ),
                  }}
                  style={{ marginBottom: '20px' }}
                />
              </div>

              {/* Summary Section */}
              <div className="summary" style={{ marginBottom: '20px' }}>
                <Typography variant="h6" className="section-title">Summary:</Typography>
                <Box className="result-box" style={boxStyle}>
                  {summary ? highlightSearchTerm(summary) : "Summary will be displayed here after analysis."}
                </Box>
              </div>

              {/* Sentiment Analysis Section */}
              <div className="analysis" style={{ marginBottom: '20px' }}>
                <Typography variant="h6" className="section-title">Sentiment Analysis:</Typography>
                <Box className="result-box" style={boxStyle}>
                  {sentiment || "Sentiment analysis result will be displayed here."}
                </Box>
              </div>

              {/* Keywords Section */}
              <div className="keywords" style={{ marginBottom: '20px' }}>
                <Typography variant="h6" className="section-title">Keywords:</Typography>
                <Box className="result-box" style={boxStyle}>
                  {keywords ? highlightSearchTerm(keywords) : "Keywords will be displayed here."}
                </Box>
              </div>

              {/* Topics Section */}
              <div className="topics" style={{ marginBottom: '20px' }}>
                <Typography variant="h6" className="section-title">Topic Modeling:</Typography>
                <Box className="result-box" style={boxStyle}>
                  {topics ? highlightSearchTerm(topics) : "Topic modeling results will be displayed here."}
                </Box>
              </div>
            </Paper>
          </Container>
        )} />
        <Route path="/" element={<LandingPage />} /> {/* Change to element prop */}
      </Routes>
    </Router>
  );
}

const boxStyle = {
  padding: '10px',
  backgroundColor: '#f5f5f5',
  borderRadius: '5px',
  minHeight: '50px',
  border: '1px solid #ddd',
};

const highlightStyle = `
  .highlight {
    background-color: yellow;
    font-weight: bold;
  }
`;

const styleSheet = document.createElement("style");
styleSheet.type = "text/css";
styleSheet.innerText = highlightStyle;
document.head.appendChild(styleSheet);

export default App;
