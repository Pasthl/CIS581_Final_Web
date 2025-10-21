// frontend/app/layout.tsx
import "./globals.css";

export const metadata = {
  title: "Denoise Web",
  description: "Minimal demo",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
