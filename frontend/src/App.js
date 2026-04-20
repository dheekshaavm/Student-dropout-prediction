import React, { useState } from "react";
import axios from "axios";
import "./App.css";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell
} from "recharts";

// Human-friendly feature names
const featureNames = [
  "School",
  "Gender",
  "Age",
  "Home address type",
  "Family size",
  "Parent relationship status",
  "Mother's education",
  "Father's education",
  "Mother's job",
  "Father's job",
  "Reason for choosing school",
  "Primary guardian",
  "Travel time to school",
  "Study time",
  "Past failures",
  "Extra school support",
  "Family support",
  "Extra paid classes",
  "Extracurricular activities",
  "Attended nursery",
  "Wants higher education",
  "Internet access",
  "Romantic relationship",
  "Family relationship quality",
  "Free time",
  "Going out frequency",
  "Weekday alcohol consumption",
  "Weekend alcohol consumption",
  "Health status",
  "Absences",
  "First period grade",
  "Second period grade"
];

function App() {
  const [form, setForm] = useState({
    age: 18,
    studytime: 2,
    failures: 0,
    absences: 5,
    G1: 10,
    G2: 12
  });

  const [result, setResult] = useState(null);

  const labels = {
    age: "Student Age",
    studytime: "Weekly Study Time",
    failures: "Past Academic Failures",
    absences: "Number of Absences",
    G1: "First Term Grade",
    G2: "Second Term Grade"
  };

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: Number(e.target.value) });
  };

  const predict = async () => {
    try {
      const res = await axios.post("http://127.0.0.1:5000/predict", form);
      setResult(res.data);
    } catch (err) {
      alert("API error");
    }
  };

  const probability = result ? (result.probability * 100).toFixed(1) : 0;
  const isRisk = result?.prediction === 1;

  // Top SHAP features
  const getTopShapFeatures = () => {
    if (!result?.shap_values) return [];
    const pairs = result.shap_values.map((value, i) => ({
      name: featureNames[i],
      value
    }));
    pairs.sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
    return pairs.slice(0, 5);
  };

  const getChartData = () =>
    getTopShapFeatures().map((f) => ({
      name: f.name,
      value: Math.abs(f.value),
      color: f.value > 0 ? "#e53935" : "#2e7d32"
    }));

  return (
    <div className="container">
      <h1 className="title">Student Dropout Risk Analyzer</h1>
      <p className="subtitle">
        Explainable AI system for early academic risk detection
      </p>

      {/* INPUT FORM */}
      <div className="card">
        <h2 className="sectionTitle">Student Information</h2>

        <div className="grid">
          {Object.keys(form).map((key) => (
            <div key={key} className="inputGroup">
              <label>{labels[key]}</label>
              <input
                type="number"
                name={key}
                value={form[key]}
                onChange={handleChange}
              />
            </div>
          ))}
        </div>

        <button className="predictBtn" onClick={predict}>
          Run Risk Analysis
        </button>
      </div>

      {result && (
        <>
          {/* RESULT */}
          <div className="card resultCard">
            <h2 className={isRisk ? "risk" : "safe"}>
              {isRisk ? "High Risk Detected" : "Low Risk Profile"}
            </h2>

            <p className="probText">
              Estimated Dropout Probability: <strong>{probability}%</strong>
            </p>

            <div className="meter">
              <div
                className="meterFill"
                style={{
                  width: `${probability}%`,
                  backgroundColor: isRisk ? "#e53935" : "#2e7d32"
                }}
              />
            </div>
          </div>

          {/* SHAP EXPLANATION */}
          <div className="card">
            <h3>🔍 Key Factors Influencing Prediction</h3>

            {getTopShapFeatures().map((feat, i) => (
              <div key={i} style={{ marginBottom: 6 }}>
                <strong>{feat.name}</strong>{" "}
                <span
                  style={{
                    color: feat.value > 0 ? "#e53935" : "#2e7d32"
                  }}
                >
                  {feat.value > 0
                    ? "increased dropout risk"
                    : "helped reduce risk"}
                </span>
              </div>
            ))}

            {/* SHAP CHART */}
            <div style={{ width: "100%", height: 260, marginTop: 20 }}>
              <ResponsiveContainer>
                <BarChart data={getChartData()}>
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="value">
                    {getChartData().map((entry, i) => (
                      <Cell key={i} fill={entry.color} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* INTERVENTIONS */}
          {result.interventions?.length > 0 && (
            <div className="card">
              <h3>🧠 Recommended Interventions</h3>
              {result.interventions.map((item, i) => (
                <div key={i} style={{ marginBottom: 6 }}>
                  ✔ {item}
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}

export default App;