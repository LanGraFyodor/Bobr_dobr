import { Card } from "@/components/Card";
import { ChartContainer } from "@/components/ChartContainer";

export const CorrelationScreen = () => {
  return (
    <div className="flex flex-col gap-6 h-full">
      <div className="flex gap-6 flex-grow">
        <div className="flex-grow flex flex-col">
          <ChartContainer title="Большая HeatMap Корреляции" className="h-full" />
        </div>
        <div className="w-80 flex flex-col gap-4">
          <Card title="Лучший результат">
            <p>Азимут: 124°</p>
            <p>Смещение: 385 м</p>
            <p>Корреляция: 0.982</p>
          </Card>
          <Card title="ТОП-5 совпадений">
            <p className="text-xs text-muted-foreground">Таблица ТОП-5...</p>
          </Card>
        </div>
      </div>
      <div className="h-64 grid grid-cols-2 gap-6">
        <ChartContainer title="Наблюдаемый профиль" className="h-full" />
        <ChartContainer title="Эталонный профиль" className="h-full" />
      </div>
    </div>
  );
};
