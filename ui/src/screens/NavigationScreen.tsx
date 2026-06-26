import { Card } from "@/components/Card";
import { MapContainer } from "@/components/MapContainer";

export const NavigationScreen = () => {
  return (
    <div className="flex gap-6 h-full">
      <div className="flex-grow flex flex-col">
        <MapContainer className="h-full flex-grow" />
      </div>
      <div className="w-80 flex flex-col gap-4">
        <Card title="Координаты">
          <p>Широта: ...</p>
          <p>Долгота: ...</p>
          <p>Высота MSL: ...</p>
          <p>Высота над рельефом: ...</p>
        </Card>
        <Card title="Движение">
          <p>Путевой угол: ...</p>
          <p>Курс: ...</p>
          <p>Путевая скорость: ...</p>
          <p>Вертикальная скорость: ...</p>
        </Card>
        <Card title="Источник позиционирования">
          <ul className="space-y-1">
            <li className="flex justify-between"><span>GNSS</span><span>✓</span></li>
            <li className="flex justify-between"><span>TERCOM</span><span>✓</span></li>
            <li className="flex justify-between"><span>INS</span><span>✓</span></li>
            <li className="flex justify-between"><span>EKF</span><span>✓</span></li>
          </ul>
        </Card>
      </div>
    </div>
  );
};
