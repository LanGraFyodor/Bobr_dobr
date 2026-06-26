import { Card } from "@/components/Card";
import { ChartContainer } from "@/components/ChartContainer";

export const SensorsScreen = () => {
  return (
    <div className="grid grid-cols-2 gap-6 h-full">
      <Card title="Радиовысотомер" className="h-[300px]">
        <p className="text-2xl font-bold mb-4">545.3 м</p>
        <ChartContainer title="График последних 20 секунд" className="h-32" />
      </Card>
      <Card title="Барометр" className="h-[300px]">
        <p>MCL</p>
        <p className="text-2xl font-bold mb-4">1123.6 м</p>
      </Card>
      <Card title="ИНС" className="h-[300px]">
        <p>Ускорение X, Y, Z...</p>
        <p>Угловые скорости: Roll, Pitch, Yaw...</p>
      </Card>
      <Card title="EKF" className="h-[300px]">
        <p>Вес INS: 0.18</p>
        <p>Вес TERCOM: 0.82</p>
        <p>Ошибка оценки (Pxx, Pyy, Pzz)...</p>
      </Card>
    </div>
  );
};
