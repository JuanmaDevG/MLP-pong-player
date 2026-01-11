#include "TPaleta.h"
#include "StdPijo.h"

////////////////////////////////////////////////////////////
// Constructor:                                           //
// * Establecemos todos los atributos por defecto. Luego  //
//   sera necesario particularizar con las funciones de   //
//   la interfaz.                                         //
////////////////////////////////////////////////////////////
TPaleta::TPaleta()
{
   // Inicializamos posicion, tamanyo, velocidad y limites
   largo = 4; panx = 1;
   y = (STDP::TamY()+4)/2; pany = (int) y; vy = 0.0;
   miny = 0; maxy = STDP::TamY() - 1;

   // Y ahora las caracteristicas de la paleta
   ch = '#'; Atr = STDP_A_NEGRITA; CT = STDP_C_BLANCO; CF = STDP_C_AZUL;
}

////////////////////////////////////////////////////////////
// Destructor:                                            //
// * Tampoco hay demasiado para destruir... xD            //
////////////////////////////////////////////////////////////
TPaleta::~TPaleta () { /* Nada que hacer */ }

////////////////////////////////////////////////////////////
// Posicionar:                                            //
// * Nos permite colocar la paleta en cualquier parte 8). //
////////////////////////////////////////////////////////////
void
TPaleta::Posicionar (int px, int py)
{
   if (px >= 0 &&  px < STDP::TamX() && py >= miny && py <= maxy)
   { panx = px; y = (float) py; pany = py; }
}

////////////////////////////////////////////////////////////
// PonVelocidad:                                          //
// * Establece la velocidad de la paleta.                 //
////////////////////////////////////////////////////////////
void
TPaleta::PonVelocidad (float valor)
{ if (valor >= -1.0 && valor <= 1.0) vy = valor; }

////////////////////////////////////////////////////////////
// PonAtributos:                                          //
// * Para cambiar el aspecto de la paletica ;-D.          //
////////////////////////////////////////////////////////////
void
TPaleta::PonAtributos (chtype pch, int pAtr, int pCT, int pCF)
{
   if (STDP::EsAtributo(pAtr)) Atr = pAtr;
   if     (STDP::EsColor(pCT)) CT = pCT;
   if     (STDP::EsColor(pCF)) CF = pCF;
   ch = pch; 
}

////////////////////////////////////////////////////////////
// PonLimites:                                            //
// * Para establecer los limites de movimiento (Asi no se //
//   nos escapa xD).                                      //
////////////////////////////////////////////////////////////
void
TPaleta::PonLimites (int pminY, int pmaxY)
{
   if (pminY < pmaxY && pminY >=0 && pmaxY < STDP::TamY())
   {
      miny = pminY; maxy = pmaxY; 
      y = (miny + maxy - largo) / 2; pany = (int) y;
   }
}

////////////////////////////////////////////////////////////
// PonLargo:                                              //
// * Con esto le decimos lo larga que es la paleta xD.    //
////////////////////////////////////////////////////////////
void
TPaleta::PonLargo (int plargo) { if (plargo <= maxy - miny) largo = plargo; }

////////////////////////////////////////////////////////////
// AplicaAceleracion:                                     //
// * Hacemos acelerar la paleta en una direccion.         //
////////////////////////////////////////////////////////////
void
TPaleta::AplicaAceleracion (float valor)
{
   // Precalculamos la nueva aceleracion
   float ry = vy + valor;

   // Y ahora comprobamos si es o no correcta y corregimos
   if (ry > 1.0)
      vy = 1.0;
   else if (ry < -1.0)
      vy = -1.0;
   else
      vy = ry;
}

////////////////////////////////////////////////////////////
// Mover:                                                 //
// * Calculamos el siguiente movimiento.                  //
////////////////////////////////////////////////////////////
void
TPaleta::Mover ()
{
   // Precalculamos la nueva posicion
   float ry = y + vy;

   // Y ahora comprobamos si es correcta y corregimos
   if (ry < miny)
   {
      y = miny; 
      vy = 0.0;
   }
   else if (ry > maxy - largo + 1)
   {
      y = maxy - largo + 1;
      vy = 0.0;
   }
   else
      y = ry;

   pany = (int)(y + 0.5); 
}

////////////////////////////////////////////////////////////
// Dibujar:                                               //
// * Con esto dibujamos nuestra paletiña.                 //
////////////////////////////////////////////////////////////
void
TPaleta::Dibujar ()
{
   int i;
   STDP::CambiaColor (Atr,CT,CF); 
   for (i=0; i<largo; i++)
   {
      STDP::PonCursor (panx, pany + i);
      STDP::EscribeCh (ch);
   }
}

////////////////////////////////////////////////////////////
// Caracter:                                              //
// * Obtenemos el caracter de la paleta.                  //
////////////////////////////////////////////////////////////
chtype
TPaleta::Caracter() const { return ch; }

////////////////////////////////////////////////////////////
// Atributo:                                              //
// * Obtenemos el atributo de la paleta.                  //
////////////////////////////////////////////////////////////
int
TPaleta::Atributo() const { return Atr; }

////////////////////////////////////////////////////////////
// ColorTexto:                                            //
// * Obtenemos el color de texto de la paleta.            //
////////////////////////////////////////////////////////////
int
TPaleta::ColorTexto() const { return CT; }

////////////////////////////////////////////////////////////
// ColorFondo:                                            //
// * Obtenemos el color de fondo de la paleta.            //
////////////////////////////////////////////////////////////
int
TPaleta::ColorFondo() const { return CF; }

////////////////////////////////////////////////////////////
// PosX:                                                  //
// * Con esto obtenemos la posicion horizontal.           //
////////////////////////////////////////////////////////////
int
TPaleta::PosX () const { return panx; }

////////////////////////////////////////////////////////////
// PosY:                                                  //
// * Con esto obtenemos la posicion vertical REAL.        //
////////////////////////////////////////////////////////////
float
TPaleta::PosY () const { return y; }

////////////////////////////////////////////////////////////
// PosY:                                                  //
// * Con esto obtenemos la velocidad actual.              //
////////////////////////////////////////////////////////////
float
TPaleta::VelY () const { return vy; }

////////////////////////////////////////////////////////////
// PosY:                                                  //
// * Con esto podemos ver lo larga que es la paleta xDD.  //
////////////////////////////////////////////////////////////
int
TPaleta::Largo () const { return largo; }
