#include "TMarcador.h"
#include <time.h>

///////////////////////////////////////////////////////////
// Constructor:                                          //
// * Mas inicializaciones nooo por favooorRr xDD.        //
///////////////////////////////////////////////////////////
TMarcador::TMarcador ()
{
   // Tiempo y puntos
   tiempo    = 0;
   tiempoIni = clock();
   tiempoOn  = true;
   puntos1 = puntos2 = 0;

   // Ventana del marcador
   minX = 0; minY = 0; maxX = STDP::TamX() - 1; maxY = STDP::TamY()/5;

   // Atributos del marco
   ch = ' '; Atr = STDP_A_NORMAL; CT = STDP_C_NEGRO; CF = STDP_C_CIAN;

   // Etiquetas
   NumEtiquetas = 3;
   Etiqueta = new TEtiqueta[NumEtiquetas];  // necesitamos minimo 3 etiquetas
   if (Etiqueta == NULL)
   {
      cerr << "No se pudo reservar memoria en TMarcador\n";
      exit (-1);
   }
   // 0 -> Tiempo, 1,2 -> Jugador 1,2
   int zx = STDP::TamX()/4;
   int zy = maxY / 3;
   Etiqueta[0].PonAtributos  (STDP_A_NEGRITA, STDP_C_MAGENTA, STDP_C_NEGRO);
   Etiqueta[0].Posicionar    (2*zx, zy);
   Etiqueta[1].PonAtributos  (STDP_A_NEGRITA, STDP_C_AZUL, STDP_C_NEGRO);
   Etiqueta[1].Posicionar    (zx, zy);
   Etiqueta[2].PonAtributos  (STDP_A_NORMAL, STDP_C_AMARILLO, STDP_C_NEGRO);
   Etiqueta[2].Posicionar    (3*zx, zy);

   for (int i=0; i < NumEtiquetas; i++) 
      Etiqueta[i].DefineVentana (minX, minY, maxX, maxY);
}

///////////////////////////////////////////////////////////
// Destructor:                                           //
// * Pos weno, no hace mucho, pero queda mono xDD.       //
///////////////////////////////////////////////////////////
TMarcador::~TMarcador () { /* 1,2,3, un pasito palante maria xDD */ }

///////////////////////////////////////////////////////////
// Inicializar:                                          //
// * Prepara el marcador pa empezar a contar.            //
///////////////////////////////////////////////////////////
void
TMarcador::Inicializar ()
{
   puntos1 = puntos2 = 0;
   PonTiempo (0.0);
}

///////////////////////////////////////////////////////////
// PonMarco:                                             //
// * Establecemos los limites de nuestro marcador.       //
///////////////////////////////////////////////////////////
void
TMarcador::PonMarco (int pminx, int pminy, int pmaxx, int pmaxy)
{
   if (pminy <= pmaxy && pminx < pmaxx && pminx >= 0 && pminy >=0 &&
       pmaxx < STDP::TamX() && pmaxy < STDP::TamY())
   { 
      minX = pminx; maxX = pmaxx; minY = pminy; maxY = pmaxy; 
      for (int i=0; i<NumEtiquetas; i++) 
         Etiqueta[i].DefineVentana (minX, minY, maxX, maxY);
   }
}

///////////////////////////////////////////////////////////
// PonTiempo:                                            //
// * Modificamos el tiempo actual.                       //
///////////////////////////////////////////////////////////
void
TMarcador::PonTiempo (unsigned long t) { tiempo = t; tiempoIni = clock(); }

///////////////////////////////////////////////////////////
// PonPuntos:                                            //
// * Modificamos el marcador actual.                     //
///////////////////////////////////////////////////////////
void
TMarcador::PonPuntos (int pnts1, int pnts2)
{ puntos1 = pnts1; puntos2 = pnts2; }

///////////////////////////////////////////////////////////
// PonAtributos:                                         //
// * Ponemos los colorines y atributo del marcador       //
///////////////////////////////////////////////////////////
void
TMarcador::PonAtributos (chtype pch, int pAtr, int pCT, int pCF)
{
   ch = pch;
   if (STDP::EsAtributo(pAtr)) Atr = pAtr;
   if     (STDP::EsColor(pCT)) CT = pCT;
   if     (STDP::EsColor(pCF)) CF = pCF;
}

///////////////////////////////////////////////////////////
// PonAtrEtiq:                                           //
// * Cambia lo colore de one etiqueta concreta.          //
///////////////////////////////////////////////////////////
void
TMarcador::PonAtrEtiq (int numetiq, int pAtr, int pCT, int pCF)
{
   if (numetiq >= 0 && numetiq < NumEtiquetas)
      Etiqueta[numetiq].PonAtributos (pAtr,pCT,pCF);
}

///////////////////////////////////////////////////////////
// PonPosEtiq:                                           //
// * Cambia la posicion de una etiqueta concreta.        //
///////////////////////////////////////////////////////////
void
TMarcador::PonPosEtiq (int numetiq, int px, int py)
{
   if (numetiq >= 0 && numetiq < NumEtiquetas)
      Etiqueta[numetiq].Posicionar (px, py);
}

///////////////////////////////////////////////////////////
// PararTiempo:                                          //
// * Detiene el paso del tiempo (mmm, que util... xD).   //
///////////////////////////////////////////////////////////
void
TMarcador::PararTiempo () { CalcularTiempo(); tiempoOn = false; }

///////////////////////////////////////////////////////////
// ContarTiempo:                                         //
// * Continua contando el paso del tiempo.               //
///////////////////////////////////////////////////////////
void
TMarcador::ContarTiempo () { tiempoOn = true; tiempoIni = clock(); }

///////////////////////////////////////////////////////////
// IncPuntos1:                                           //
// * Incrementa los puntos del jugador 1.                //
///////////////////////////////////////////////////////////
int
TMarcador::IncPuntos1() { return ++puntos1; }

///////////////////////////////////////////////////////////
// IncPuntos2:                                           //
// * I'm a slave for you...                              //
///////////////////////////////////////////////////////////
int
TMarcador::IncPuntos2() { return ++puntos2; }

///////////////////////////////////////////////////////////
// Dibujar:                                              //
// * Lalaralarita, pinto mi casita... xDDD.              //
///////////////////////////////////////////////////////////
void
TMarcador::Dibujar()
{
   int y = (minY + maxY) / 2;
   int x = (minX + maxX) / 4;
   int mins, secs;
   char cad[24];

   // Calculamos el tiempo actual (si estamos On)
   if (tiempoOn) CalcularTiempo();
   secs = tiempo / CLOCKS_PER_SEC;
   mins = secs / 60;
   secs = secs % 60;

   // Dibujamos el marco
   STDP::CambiaColor   (Atr, CT, CF);
   STDP::DibujaCuadrado(minX, minY, maxX, maxY, (chtype) ch);
   STDP::PonCursor     (x,y);

   // Preparamos las etiquetas y las imprimimos
   sprintf (cad, "Goles: %i", puntos1);
   Etiqueta[1].PonCadena (cad);
   sprintf (cad, "Tiempo: %im %is", mins, secs);
   Etiqueta[0].PonCadena (cad);
   sprintf (cad, "Goles: %i", puntos2);
   Etiqueta[2].PonCadena (cad);

   for (int i = 0; i < NumEtiquetas; i++) Etiqueta[i].Dibujar();
}

///////////////////////////////////////////////////////////
// Tiempo:                                               //
// * Devolvemos el tiempo de juego (en segundos).        //
///////////////////////////////////////////////////////////
unsigned long
TMarcador::Tiempo () const { return (tiempo/CLOCKS_PER_SEC); }

///////////////////////////////////////////////////////////
// Puntos1:                                              //
// * Devuelve los puntos del jugador 1.                  //
///////////////////////////////////////////////////////////
int
TMarcador::Puntos1 () const { return puntos1; }

///////////////////////////////////////////////////////////
// Puntos2:                                              //
// * Devuelve los puntos del jugador 2.                  //
///////////////////////////////////////////////////////////
int
TMarcador::Puntos2 () const { return puntos2; }

///////////////////////////////////////////////////////////
// AtributoEtiq:                                         //
// * Devuelve el atributo de la etiqueta (-1 si no exst) //
///////////////////////////////////////////////////////////
int
TMarcador::AtributoEtiq (int numetiq) const 
{
   if (numetiq >= 0 && numetiq<NumEtiquetas)
      return Etiqueta[numetiq].Atributo();
   else
      return -1; 
}

///////////////////////////////////////////////////////////
// ColorTextoEtiq:                                       //
// * Devuelve el Col.Txt. de la etiqueta (-1 si no exst) //
///////////////////////////////////////////////////////////
int
TMarcador::ColorTextoEtiq (int numetiq) const 
{
   if (numetiq >= 0 && numetiq<NumEtiquetas)
      return Etiqueta[numetiq].ColorTexto();
   else
      return -1; 
}

///////////////////////////////////////////////////////////
// ColorFondoEtiq:                                       //
// * Devuelve el Col.Fnd. de la etiqueta (-1 si no exst) //
///////////////////////////////////////////////////////////
int
TMarcador::ColorFondoEtiq (int numetiq) const 
{
   if (numetiq >= 0 && numetiq<NumEtiquetas)
      return Etiqueta[numetiq].ColorFondo();
   else
      return -1; 
}

///////////////////////////////////////////////////////////
// CalcularTiempo:                                       //
// * Calcula el tiempo actual y actualiza.               //
///////////////////////////////////////////////////////////
void
TMarcador::CalcularTiempo ()
{
   unsigned long tmp = clock();
   tiempo += tmp - tiempoIni;
   tiempoIni = tmp;
}
