import { useState, useRef } from 'react';
import { UploadCloud, FileType, CheckCircle2, AlertTriangle, RefreshCw } from 'lucide-react';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const analyzeBinary = async () => {
    if (!file) return;
    
    setIsAnalyzing(true);
    setError(null);
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      const response = await fetch('http://localhost:8000/analyze', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || 'Analysis failed');
      }
      
      const data = await response.json();
      setResults(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const reset = () => {
    setFile(null);
    setResults(null);
    setError(null);
  };

  return (
    <div className="app-container">
      <header className="header">
        <h1>CryptoTrace RT</h1>
        <p>Real-Time Cryptographic Binary Detection via Micro-Architectural Profiling</p>
      </header>

      {!results && !isAnalyzing && (
        <div 
          className={`glass-panel upload-zone ${isDragging ? 'drag-active' : ''}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <UploadCloud className="upload-icon" />
          <h2>{file ? file.name : 'Drop an ELF binary here'}</h2>
          <p style={{ marginTop: '10px', color: 'var(--text-muted)' }}>
            or click to browse your files
          </p>
          <input 
            type="file" 
            ref={fileInputRef} 
            onChange={handleFileChange} 
            className="file-input"
          />
          
          {file && (
            <button 
              className="glass-button" 
              style={{ marginTop: '30px' }}
              onClick={(e) => { e.stopPropagation(); analyzeBinary(); }}
            >
              Analyze Binary
            </button>
          )}
        </div>
      )}

      {isAnalyzing && (
        <div className="glass-panel loading-container">
          <div className="spinner"></div>
          <h2>Analyzing Binary...</h2>
          <p style={{ marginTop: '10px', color: 'var(--text-muted)' }}>
            Extracting LIEF semantics and executing hardware profiling harness...
          </p>
        </div>
      )}

      {error && (
        <div className="glass-panel" style={{ padding: '20px', textAlign: 'center', borderColor: 'var(--secondary)' }}>
          <AlertTriangle color="var(--secondary)" size={48} style={{ marginBottom: '15px' }} />
          <h3 style={{ color: 'var(--secondary)' }}>Analysis Error</h3>
          <p>{error}</p>
          <button className="glass-button" style={{ marginTop: '20px' }} onClick={reset}>Try Again</button>
        </div>
      )}

      {results && (
        <>
          <div className="dashboard-grid">
            <div className="glass-panel result-card">
              <h3>Classification Result</h3>
              <div className={`prediction-badge ${results.prediction === 'Crypto' ? 'prediction-crypto' : 'prediction-noncrypto'}`}>
                {results.prediction}
              </div>
              <p>Confidence Score: {(results.confidence * 100).toFixed(2)}%</p>
              
              <div className="confidence-meter">
                <div 
                  className="confidence-fill" 
                  style={{ 
                    width: `${results.confidence * 100}%`,
                    background: results.prediction === 'Crypto' ? 'var(--secondary)' : 'var(--success)'
                  }}
                ></div>
              </div>
              
              <button className="glass-button" style={{ marginTop: '40px' }} onClick={reset}>
                <RefreshCw size={18} style={{ marginRight: '8px', verticalAlign: 'middle' }}/>
                Analyze Another
              </button>
            </div>
            
            <div className="glass-panel shap-card">
              <h3>Top 10 Discriminative Features (SHAP)</h3>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '20px' }}>
                Features pushing right (Red) drive the model towards Crypto. Features pushing left (Cyan) drive towards Non-Crypto.
              </p>
              
              <div className="shap-list">
                {results.top_features_shap.slice(0,10).map((f, i) => {
                  const maxShap = Math.max(...results.top_features_shap.map(x => Math.abs(x.shap_value)));
                  const width = (Math.abs(f.shap_value) / maxShap) * 50; // max 50% width from center
                  const isPositive = f.shap_value > 0;
                  
                  return (
                    <div key={i} className="shap-item">
                      <div className="shap-label" title={`Raw Value: ${f.raw_value.toFixed(4)}`}>
                        {f.feature}
                      </div>
                      <div className="shap-bar-container">
                        <div 
                          className={`shap-bar ${isPositive ? 'positive' : 'negative'}`}
                          style={{ width: `${width}%` }}
                        ></div>
                      </div>
                      <div className="shap-value" style={{ color: isPositive ? 'var(--secondary)' : 'var(--primary)' }}>
                        {f.shap_value > 0 ? '+' : ''}{f.shap_value.toFixed(3)}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
          
          <div className="glass-panel features-table-container">
            <h3 style={{ marginBottom: '20px', color: 'var(--primary)' }}>Extracted Feature Telemetry</h3>
            <table className="features-table">
              <thead>
                <tr>
                  <th>Feature Name</th>
                  <th>Extracted Value</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(results.all_features).map(([key, value]) => (
                  <tr key={key}>
                    <td>{key}</td>
                    <td>{Number(value).toFixed(6)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}

export default App;
