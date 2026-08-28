import { useEffect, useState } from "react";
import { checkBackendHealth } from "./api/health";

function App() {
  const [backendStatus, setBackendStatus] = useState("Checking...");
  const [error, setError] = useState("");

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const data = await checkBackendHealth();

        if (data.status === "healthy") {
          setBackendStatus("Backend Connected");
        } else {
          setBackendStatus("Backend Running");
        }
      } catch {
        setBackendStatus("Backend Disconnected");
        setError("Unable to connect to FastAPI backend.");
      }
    };

    checkHealth();
  }, []);

  return (
    <main>
      <h1>GeoMineral AI</h1>

      <p>AI-Powered Mineral Prospectivity Mapping</p>

      <section>
        <h2>System Status</h2>

        <p>{backendStatus}</p>

        {error && <p>{error}</p>}
      </section>
    </main>
  );
}

export default App;