import { Header } from "./components/Header";
import { StatusBar } from "./components/StatusBar";
import { Tabs } from "./components/Tabs";
import { NavigationScreen } from "./screens/NavigationScreen";
import { CorrelationScreen } from "./screens/CorrelationScreen";
import { SensorsScreen } from "./screens/SensorsScreen";
import { PerformanceScreen } from "./screens/PerformanceScreen";
import { SettingsScreen } from "./screens/SettingsScreen";

function App() {
  const tabs = [
    { id: "nav", label: "Навигация", content: <NavigationScreen /> },
    { id: "corr", label: "Корреляция", content: <CorrelationScreen /> },
    { id: "sensors", label: "Датчики", content: <SensorsScreen /> },
    { id: "perf", label: "Производительность", content: <PerformanceScreen /> },
    { id: "settings", label: "Настройки и симуляция", content: <SettingsScreen /> },
  ];

  return (
    <div className="flex flex-col h-screen w-screen overflow-hidden bg-background text-foreground">
      <Header />
      <main className="flex-grow flex overflow-hidden">
        <Tabs tabs={tabs} />
      </main>
      <StatusBar />
    </div>
  );
}

export default App;
