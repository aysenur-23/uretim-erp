import type { Metadata } from "next";
import { Geist } from "next/font/google";
import "./globals.css";

const geist = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Mega Insulation - Kalite Kontrol",
  description: "Taş yünü ürünler için görsel tabanlı kalite kontrol sistemi.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="tr" className={`${geist.variable} antialiased`}>
      <body>{children}</body>
    </html>
  );
}
