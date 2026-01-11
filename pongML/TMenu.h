// Clase para hacer menus
#ifndef __TMENU_H_
#define __TMENU_H_

#include "StdPijo.h"
#include "TEtiqueta.h"
#include <time.h>

// Modos de fondo
//--------------------------
#define TMENU_MODO_PLANO   0
#define TMENU_MODO_NIEVE   1

class TMenu
{
public:
   // Constructor y Destructor
   TMenu  (int pnumOpciones = 4);
   ~TMenu ();
   
   // Funciones Set
   void PonAtrOpcion    (int numopc, int Atr, int CT, int CF);
   void PonAtrTitulo    (int Atr, int CT, int CF);
   void PonPosOpcion    (int numopc, int x, int y);
   void PonPosTitulo    (int x, int y);
   void PonAtributos    (int Atr, int CT, int CF);
   void PonCadenaOpcion (int numopc, const char *cad);
   void PonCadenaTitulo (const char *cad);
   void PonSeleccion    (int numopc);
   void DefineVentana   (int minX, int minY, int maxX, int maxY);
   void PonMarco        (const char *cad, int FPS, int Atr, int CT, int CF);
   void PonFondo        (int pmodo, int pcolor);
   
   // Ejecucion
   int  Ejecutar (double FPS);

private:
   int numOpciones{};              // Numero de opciones del menu
   int seleccion{};                // Opcion seleccionada
   TEtiqueta *opcion{};            // Opciones
   TEtiqueta titulo{};             // Titulo del menu
   int minX{}, minY{}, maxX{}, maxY{};   // Ventana del menu
   struct mm 
   {
      char *cadena{};             
      int Atr{}, CT{}, CF{};
      int largo{};
      int FPS{};
      bool activo{};
   } marco;                      // Marco de texto
   int modo{};                     // Modo del fondo (plano, efecto...)
   int color{};                    // Color del fondo
      
   void Dibujar();               // Dibujado del menu
};

#endif