// La paletica...
#ifndef __TPALETA_H_
#define __TPALETA_H_
#include "StdPijo.h"

class TPaleta
{
public:
   // Constructor y destructor...
   TPaleta ();
   ~TPaleta ();

   // Funciones Set
   void  Posicionar        (int px, int py);
   void  PonAtributos      (chtype pch, int pAtr, int pCT, int pCF);
   void  PonLimites        (int pminY, int pmaxY);
   void  PonLargo          (int plargo);
   void  PonVelocidad      (float valor);
   void  AplicaAceleracion (float valor);

   // Movimiento y dibujado
   void  Mover ();
   void  Dibujar ();

   // Funciones Get
   chtype Caracter() const;
   int   Atributo() const;
   int   ColorTexto() const;
   int   ColorFondo() const;
   int   PosX () const;
   float PosY () const;
   int   Largo () const;
   float VelY() const;

private:
   float y;              // Posicion superior de la paleta
   float vy;             // Velocidad actual de la paleta
   int   panx, pany;     // Posicion de la paleta en pantalla 
   int   maxy, miny;     // Posiciones maximas superior e inferior de la paleta
   int   largo;          // Longitud de la paleta
   char  ch;             // caracter de la paleta
   int   Atr, CT, CF;    // Atributos de la paleta
};
#endif
