import { Card } from "@/components/Card";
import { ChartContainer } from "@/components/ChartContainer";

export const PerformanceScreen = () => {
  return (
    <div className="flex flex-col gap-6 h-full">
      <div className="grid grid-cols-4 gap-6">
        <Card title="Время цикла">
          <p className="text-2xl font-bold">14.8 мс</p>
        </Card>
        <Card title="FPS">
          <p className="text-2xl font-bold">67</p>
        </Card>
        <Card title="CPU">
          <p className="text-2xl font-bold">34 %</p>
        </Card>
        <Card title="RAM">
          <p className="text-2xl font-bold">512 МБ</p>
        </Card>
      </div>
      <div className="flex-grow">
        <ChartContainer title="Графики производительности" className="h-full" />
      </div>
    </div>
  );
};
