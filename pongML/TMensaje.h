// Clase para mostrar mensajes en pantalla
#ifndef __TMENSAJE_H_
#define __TMENSAJE_H_

#include "StdPijo.h"
#include "TEtiqueta.h"

// Constantes de tipo de cuadro de mensaje
//----------------------------------------
#define TMENSAJE_T_OK      0
#define TMENSAJE_T_SINO    1

// Constantes de tipo de respuesta elegida
//----------------------------------------
#define TMENSAJE_R_NO      0
#define TMENSAJE_R_SI      1
#define TMENSAJE_R_OK      2

class TMensaje
{
public:
   // Constructor y destructor
   TMensaje  ();   
   ~TMensaje ();
   
   // Funciones Set
   void PonTitulo    (const char *tit);
   void PonTexto     (const char *txt);
   void PonAtrTitulo (int Atr, int CT, int CF);
   void PonAtrTexto  (int Atr, int CT, int CF);
   void PonTipo      (int tipo);
   void PonMarco     (chtype, int Atr, int CT, int CF);
   void PonFondo     (chtype, int Atr, int CT, int CF);
   void PonAncho     (int anchomens);
   void PonSombra    ();
   void QuitaSombra  ();
   
   // Ejecucion
   int Ejecutar   ();

private:
   TEtiqueta texto;        // Texto del mensaje a mostrar
   TEtiqueta titulo;       // Titulo para la ventana del mensaje
   int  tipoMensaje;       // Tipo de mensaje (OK, Si o No, etc...)
   int  minX,minY,
        maxX,maxY;         // Tamaño de la ventana del mensaje
   chtype chM;             // Caracter del marco
   int AtrM, CTM, CFM;     // Atributos del marco
   chtype chF;             // Caracter del fondo
   int AtrF, CTF, CFF;     // Atributos del fondo
   bool sombra;            // Indica si la ventana tiene sombra o no
};

#endif