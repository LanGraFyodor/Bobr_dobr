import React, { useState } from "react";
import { cn } from "@/lib/utils";

export const Tabs = ({ tabs }: { tabs: { id: string, label: string, content: React.ReactNode }[] }) => {
  const [activeTab, setActiveTab] = useState(tabs[0].id);

  return (
    <div className="flex flex-col h-full w-full">
      <div className="flex border-b bg-muted/20 px-4">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={cn(
              "px-4 py-2 text-sm font-medium border-b-2 transition-colors",
              activeTab === tab.id 
                ? "border-primary text-foreground" 
                : "border-transparent text-muted-foreground hover:text-foreground"
            )}
          >
            {tab.label}
          </button>
        ))}
      </div>
      <div className="flex-grow p-6 overflow-auto">
        {tabs.find(t => t.id === activeTab)?.content}
      </div>
    </div>
  );
};
