// Etiquetitas

#ifndef __TETIQUETA_H_
#define __TETIQUETA_H_

#include "StdPijo.h"
#include <stdlib.h>

class TEtiqueta
{
public:
   // Constructor y destructor
   TEtiqueta (const char *str = NULL);
   ~TEtiqueta();

   // Funciones Set
   void PonCadena     (const char *str);
   void Posicionar    (int px, int py);
   void DefineVentana (int pminx, int pminy, int pmaxx, int pmaxy);
   void PonAtributos  (int Atr, int CT, int CF);

   // Dibujado
   void Dibujar ();   

   // Funciones Get
   char *Cadena () const;
   int  Tamanyo () const;
   int  PosX () const;
   int  PosY () const;
   int  Atributo () const;
   int  ColorTexto () const;
   int  ColorFondo () const;
private:
   char *cadena{};     // Cadena
   int  tamanyo{};     // Tama�o de la cadena
   int  x{}, y{};        // Posicion de la cadena en pantalla
   int  minX{}, minY{},
        maxX{}, maxY{};  // Ventana que engloba la etiqueta
   int  Atr{}, CT{}, CF{}; // Atributos y colores de la cadena
};
#endif