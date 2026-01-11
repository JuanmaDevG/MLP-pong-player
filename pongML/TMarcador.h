// Marcador chachiguay :P
#ifndef __TMARCADOR_H_
#define __TMARCADOR_H_

#include "StdPijo.h"
#include "TEtiqueta.h"
#include <stdio.h>

class TMarcador
{
public:
   // Constructor y destructor
   TMarcador  ();
   ~TMarcador ();
   
   // Funciones Set
   void Inicializar  ();
   void PonMarco     (int pminX, int pminY, int pmaxX, int pmaxY);
   void PonTiempo    (unsigned long t);
   void PonPuntos    (int pnts1, int pnts2);
   void PonAtributos (chtype pch, int pAtr, int pCT, int pCF);
   void PonAtrEtiq   (int numetiq, int pAtr, int pCT, int pCF);
   void PonPosEtiq   (int numetiq, int px, int py);
   void PararTiempo  ();
   void ContarTiempo ();
   int  IncPuntos1   ();
   int  IncPuntos2   ();

   // Dibujado
   void Dibujar ();

   // Funciones Get
   unsigned long Tiempo() const;
   int  Puntos1() const;
   int  Puntos2() const;
   int  AtributoEtiq(int numetiq) const;
   int  ColorTextoEtiq(int numetiq) const;
   int  ColorFondoEtiq(int numetiq) const;
private:
   int puntos1, puntos2;         // Puntuacion de ambos jugadores
   unsigned long tiempo;         // Tiempo transcurrido (segundos)
   long tiempoIni;               // Tiempo inicial (clocks)
   bool tiempoOn;                // Indica si el tiempo esta o no en marcha
   int minX, minY, maxX, maxY;   // Marco que encierra el marcador
   chtype ch;                    // Caracter de fondo
   int Atr, CT, CF;              // Atributos del fondo
   TEtiqueta *Etiqueta;          // Etiquetas del marcador   
   int NumEtiquetas;             // Numero de etiquetas en el marcador

   void CalcularTiempo();        // Calcula el tiempo actual
};
#endif
