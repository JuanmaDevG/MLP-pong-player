#ifndef __TPELOTA_H_
#define __TPELOTA_H_

#include "TPaleta.h"
#include "StdPijo.h"

// Nuestra pelotita
class TPelota
{
public:
   // Constructor y destructor
   TPelota();
   ~TPelota();

   // Funciones Set
   void   Crear(float pvx=-2.0, float pvy=-2.0, int px=-1, int py=-1);
   void   Posicionar   (int px, int py);
   void   PonVelocidad (float vx, float vy);
   void   PonAtributos (chtype pch, int pAtr, int pCT, int PCF);
   void   DefineVentana (int minx, int miny, int maxx, int maxy);

   // Movimiento y dibujado
   int    Mover();
   void   Dibujar() const;
   bool   CompruebaChoque (const TPaleta &paleta);
  
   // Funciones Get
   float  PosX () const;
   float  PosY () const;
   float  VelX () const;
   float  VelY () const;
   chtype Caracter() const;
   int    Atributo() const;
   int    ColorTexto() const;
   int    ColorFondo() const;

private:
   int   panx, pany;   // Posicion de la pelota en pantalla
   float x, y;         // Posicion real de la pelota
   float vx, vy;       // Velocidad de la pelota 
   chtype ch;          // Caracter de la pelota
   chtype chF;	        // Caracter del fondo
   int   Atr,CT,CF;    // Atributo, color texto y color fondo de la pelota
   int   AtrF,CTF,CFF; // Atributos, color texto y color fondo del fondo
   int   minX, minY, 
         maxX, maxY;   // Ventana de movimiento de la pelota
};
#endif
