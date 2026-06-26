import React from "react";
import { cn } from "@/lib/utils";

export const Card = ({ title, children, className }: { title: string, children: React.ReactNode, className?: string }) => {
  return (
    <div className={cn("rounded-lg border bg-card text-card-foreground shadow-sm p-4 flex flex-col gap-2", className)}>
      <h3 className="font-semibold leading-none tracking-tight">{title}</h3>
      <div className="text-sm text-muted-foreground">{children}</div>
    </div>
  );
};
