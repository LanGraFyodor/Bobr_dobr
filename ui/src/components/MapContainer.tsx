import { cn } from "@/lib/utils";

export const MapContainer = ({ className }: { className?: string }) => {
  return (
    <div className={cn("border-2 border-dashed border-gray-300 rounded-lg bg-slate-100 flex items-center justify-center min-h-[400px]", className)}>
      <div className="text-center">
        <p className="text-lg font-medium text-gray-500">Контейнер карты (MapLibre GL)</p>
        <p className="text-sm text-gray-400">Отображение маршрута и области поиска</p>
      </div>
    </div>
  );
};
