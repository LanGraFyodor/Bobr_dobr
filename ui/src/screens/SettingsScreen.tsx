import { Card } from "@/components/Card";

export const SettingsScreen = () => {
  return (
    <div className="grid grid-cols-2 gap-6 h-full">
      <div className="flex flex-col gap-6">
        <Card title="Алгоритм">
          <div className="space-y-2">
            <p>Размер окна: 100</p>
            <p>Шаг азимута: 1°</p>
            <p>Максимальный радиус: 5000 м</p>
          </div>
        </Card>
        <Card title="Цифровая модель">
          <div className="space-y-2">
            <label className="flex items-center gap-2"><input type="radio" name="dem" defaultChecked /> Copernicus</label>
            <label className="flex items-center gap-2"><input type="radio" name="dem" /> FABDEM</label>
            <label className="flex items-center gap-2"><input type="radio" name="dem" /> SRTM</label>
          </div>
        </Card>
      </div>
      <div className="flex flex-col gap-6">
        <Card title="Симуляция (Чекбоксы)">
          <div className="space-y-2">
            <label className="flex items-center gap-2"><input type="checkbox" /> Потеря GNSS</label>
            <label className="flex items-center gap-2"><input type="checkbox" /> Джамминг</label>
            <label className="flex items-center gap-2"><input type="checkbox" /> Ошибка барометра</label>
            <label className="flex items-center gap-2"><input type="checkbox" /> Шум радиовысотомера</label>
          </div>
        </Card>
        <Card title="Управление">
          <div className="flex gap-4">
            <button className="bg-primary text-primary-foreground px-4 py-2 rounded">Старт</button>
            <button className="bg-destructive text-destructive-foreground px-4 py-2 rounded">Стоп</button>
            <button className="bg-secondary text-secondary-foreground px-4 py-2 rounded">Сброс</button>
          </div>
        </Card>
      </div>
    </div>
  );
};
