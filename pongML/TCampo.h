// Aqui definimos las propiedades del campo de juego
#ifndef __TCAMPO_H__
#define __TCAMPO_H__

#include "StdPijo.h"
#include "TPelota.h"
#include "TPaleta.h"
#include "TMarcador.h"

// Direccion de cada porteria
#define PORTERIA_IZQ    -1
#define PORTERIA_DCHA   1

class TCampo
{
public:
   // Constructor y destructor
   TCampo  ();
   ~TCampo ();
   
   // Funciones Set
   void  Inicializar  ();
   void  PonLimites   (int pminX, int pminY, int pmaxX, int pmaxY);
   void  PosPelota    (int px, int py);
   void  TipoPelota   (chtype pch, int pAtr, int pCT, int pCF);
   void  TipoPaletas  (chtype pch1, int pAtr1, int pCT1, int pCF1,
                       chtype pch2, int pAtr2, int pCT2, int pCF2);
   void  LargoPaletas (int largoizq, int largodcha);
   void  VelPaletas   (float vp1, float vp2);
   void  TipoCampo    (chtype pch, int pAtr, int pCT, int pCF);
   
   // Dibujar y mover
   void  Dibujar ();     
   int   Mover   (float AcelIzq, float AcelDcha);
   void  Sacar   (int Direccion);

   // Funciones Get
   int   Atributo       () const;
   int   ColorTexto     () const;
   int   ColorFondo     () const;
   int   CentroX        () const;
   int   CentroY        () const;
   int   LimiteMaxY     () const;
   int   LimiteMinY     () const;
   float PaletaIzqX     () const { return JugIzq->PosX(); }
   float PaletaIzqY     () const;
   float PaletaDchaX    () const { return JugDcha->PosX(); }
   float PaletaDchaY    () const;
   float PaletaIzqVel   () const;
   float PaletaDchaVel  () const;
   int   PaletaIzqLargo () const;
   int   PaletaDchaLargo() const;
   float PelotaX        () const;
   float PelotaY        () const;
   float PelotaVX       () const { return Pelota->VelX(); }
   float PelotaVY       () const { return Pelota->VelY(); }
private:
   int     minX, maxX, minY, maxY;  // Limites del terreno de juego
   chtype  ch;                      // Caracter de fondo
   int     Atr, CT, CF;             // Color de fondo
   TPelota *Pelota;                 // Pelota de juego
   TPaleta *JugIzq, *JugDcha;       // Los dos jugadores
};

#endif
