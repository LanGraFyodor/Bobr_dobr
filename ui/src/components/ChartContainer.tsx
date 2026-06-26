import { cn } from "@/lib/utils";

export const ChartContainer = ({ title, className }: { title: string, className?: string }) => {
  return (
    <div className={cn("border border-dashed rounded-lg bg-muted/30 flex items-center justify-center p-8", className)}>
      <p className="text-muted-foreground">[{title}] - Здесь будет график Plotly</p>
    </div>
  );
};
