export const StatusBar = () => {
  return (
    <footer className="border-t bg-muted/50 px-6 py-2 flex items-center gap-6 text-sm text-muted-foreground mt-auto">
      <div><span className="font-semibold">GNSS:</span> Ожидание</div>
      <div><span className="font-semibold">TERCOM:</span> Ожидание</div>
      <div><span className="font-semibold">INS:</span> Ожидание</div>
      <div><span className="font-semibold">EKF:</span> Ожидание</div>
    </footer>
  );
};
