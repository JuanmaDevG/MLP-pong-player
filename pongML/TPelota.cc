#include "TPelota.h"
#include "TPaleta.h"
#include "StdPijo.h"
#include <math.h>
#include <stdlib.h>

const float g_MaxVx = 1.00f;
const float g_MinVx = 0.40f;
const float g_MaxVy = 1.00f;
const float g_MinVy = 0.06f;

//////////////////////////////////////////////////////////////////
// Constructor:                                                 //
// * Pone la pelota con parametros por defecto. Mas tarde,      //
//   estos pueden ser modificados con las distintas funciones   //
//   de la interfaz.                                            //
//////////////////////////////////////////////////////////////////
TPelota::TPelota ()
{
   // Atributos, colores y caracteres
   Atr  = STDP_A_NORMAL; CT  = STDP_C_ROJO;  CF  = STDP_C_NEGRO; ch  = 'O';

   // Tamanyo de la ventana inicial
   minX = minY = 0;
   maxX = STDP::TamX() - 1; maxY = STDP::TamY() - 1;

   // Creamos la pelota en el centro con velocidad aleatoria
   // (Ojito! No vayamos a crear la pelota antes de definir los
   //  limites del campo, que podria ser un poco desastrosillo xD)
   Crear();

}
//////////////////////////////////////////////////////////////////
// Destructor:                                                  //
// * Digamos que no hay mucho que destruir... xD                //
//////////////////////////////////////////////////////////////////
TPelota::~TPelota ()  { /* Nada que hacer */ }

//////////////////////////////////////////////////////////////////
// Crear:                                                       //
// * Creacion de una nueva pelota en el centro de la ventana    //
//   con velocidad aleatoria. Tambien es posible proporcionar   //
//   posicion y velocidad mediante parametros.                  //
//////////////////////////////////////////////////////////////////
void
TPelota::Crear(float pvx, float pvy, int px, int py)
{
   float zx = 0, zy = 0;

   // Solo posicionamos por parametro con ambas coordenadas
   if (px >= minX && px <= maxX && py >= minY && py <= maxY)
      Posicionar (px, py);
   else
      Posicionar ((maxX + minX)/2, (maxY + minY)/2);

   // Para las velocidades, nos sobra con 1 proporcionada,
   // la segunda podria ser aleatoria.
   if (pvx >= -1.0 && pvx <= 1.0)
      zx = pvx;
   else
      zx = (rand() % 20 - 10) / 10.0;

   if (pvy >= -1.0 && pvy <= 1.0)
      zy = pvy;
   else {
      //const unsigned int range = static_cast<unsigned int>(g_MinVy * 100 * 4);
      //zy = (rand() % (2*range) - range) / 100.0;
      constexpr int range = 20;
      zy = (rand() % (2*range) - range) / 100.0;
   }

   PonVelocidad (zx, zy);
}

//////////////////////////////////////////////////////////////////
// Posicionar:                                                  //
// * Metodo para cambiar la posicion de la pelota de un plumazo //
//////////////////////////////////////////////////////////////////
void
TPelota::Posicionar (int px, int py)
{
   if (px >= minX && px <= maxX) { x = px; panx = px; }
   if (py >= minY && py <= maxY) { y = py; pany = py; }
}

//////////////////////////////////////////////////////////////////
// PonVelocidad:                                                //
// * Este es pa poner la velocidad que quieras a la bola.       //
//////////////////////////////////////////////////////////////////
void
TPelota::PonVelocidad (float pvx, float pvy)
{
   vx=vy=0;
   if (pvx >= -1.0 && pvx <= 1.0) vx = pvx;
   if (pvy >= -1.0 && pvy <= 1.0) vy = pvy;
   if (vx >= 0.0 && vx <  g_MinVx) vx =  g_MinVx;
   if (vx  < 0.0 && vx > -g_MinVx) vx = -g_MinVx;
   if (vy >= 0.0 && vy <  g_MinVy) vy =  g_MinVy;
   if (vy  < 0.0 && vy > -g_MinVy) vy = -g_MinVy;
}

//////////////////////////////////////////////////////////////////
// PonAtributos:                                                //
// * Pa cambia la forma y colorines de la pelotica...           //
//////////////////////////////////////////////////////////////////
void
TPelota::PonAtributos (chtype pch, int pAtr, int pCT, int pCF)
{
   if (STDP::EsAtributo(pAtr)) Atr = pAtr;
   if (STDP::EsColor(pCT)) CT = pCT;
   if (STDP::EsColor(pCF)) CF = pCF;
   ch = pch;
}

//////////////////////////////////////////////////////////////////
// DefineVentana:                                               //
// * Define la ventana de movimiento valida para la pelota.     //
//////////////////////////////////////////////////////////////////
void
TPelota::DefineVentana (int minx, int miny, int maxx, int maxy)
{
   if (minx < maxx && miny < maxy && minx >=0 && miny >=0 &&
       maxx < STDP::TamX() && maxy < STDP::TamY())
   {  minX = minx; minY = miny; maxX = maxx; maxY = maxy;  }
}

//////////////////////////////////////////////////////////////////
// Dibujar:                                                     //
// * Con esto pintamos la pelotinya...                          //
//////////////////////////////////////////////////////////////////
void
TPelota::Dibujar () const
{
   STDP::CambiaColor (Atr,CT,CF);
   STDP::PonCursor   (panx, pany);
   STDP::EscribeCh   (ch);
}

//////////////////////////////////////////////////////////////////
// Mover:                                                       //
// * Control de los movimientos de la pelota                    //
//////////////////////////////////////////////////////////////////
int
TPelota::Mover ()
{
   int Choque = 0;      // Para saber si hay gol en un lado de la pista
   float rx = x + vx;   // Calculo provisional de la nueva posicion
   float ry = y + vy;

   // Comprobamos el choque con los bordes laterales
   if      (rx < minX) { Choque = -1; x = minX; }
   else if (rx > maxX) { Choque = 1; x = maxX; }
                  else x = rx;

   // Y ahora el choque con las paredes superior e inferior
   if (ry < minY)
   {
      y = minY;
      vy = -vy;
   }
   else if (ry > maxY)
   {
      y = maxY;
      vy = -vy;
   }
   else
      y = ry;

   // La posicion en pantalla se obtiene redondeando...
   panx = (int)(x + 0.5);
   pany = (int)(y + 0.5);

   // Devolvemos el choque
   return (Choque);
}

//////////////////////////////////////////////////////////////////
// CompruebaChoque (TPaleta):                                   //
// * Comprueba si ha chocado con una paleta y reacciona.        //
//////////////////////////////////////////////////////////////////
bool
TPelota::CompruebaChoque (const TPaleta &Pal)
{
   int    palX  = Pal.PosX();           // Posicion X de la paleta
   float fpalY  = Pal.PosY();           // Posicion Y REAL de la paleta
   int    palY  = (int)(fpalY + 0.5);   // Posicion Y en pantalla de la paleta
   float  vpalY = Pal.VelY();           // Velocidad Y de la paleta
   int    largoPal = Pal.Largo();       // Largo de la paleta
   bool   contacto = false;

   // Primero vemos si hay contacto
   if (palX == panx && pany >= palY && pany < palY + largoPal)
   {
      // Todo este codigo ya rula en condiciones, asi que
      // cortare los dedos a aquel que lo modifique xDDD
      float signoyPelota = (vy == 0.0) ? 0 : vy / fabs(vy);
      float signoxPelota;
      float distMaxima = ((float)largoPal) / 2.0;
      float distCentro = y - (fpalY + distMaxima);

      // Primero hay que invertir la direccion
      vx = -vx;
      signoxPelota = vx / fabs (vx);

      // Calculamos la modificacion de velocidad Y
      vy += vpalY / 2.0;
      if (vy >=0) {
         if       (vy >  g_MaxVy) vy =  g_MaxVy;
         else if  (vy <  g_MinVy) vy =  g_MinVy;
      } else if (vy < 0) {
         if       (vy < -g_MaxVy) vy = -g_MaxVy;
         else if  (vy > -g_MinVy) vy = -g_MinVy;
      }

      // Y ahora la modificadion de velocidad X
      if (vx > 0.0)
      {
         vx += signoyPelota * (vpalY / 5.0);
         vx += signoyPelota * 0.5 * (distCentro / distMaxima);
         if (vx < g_MinVx) vx = g_MinVx;
         else if (vx > g_MaxVx) vx = g_MaxVx;
      }
      else
      {
         vx -= signoyPelota * (vpalY / 5.0);
         vx -= signoyPelota * 0.5 * (distCentro / distMaxima);
         if (vx > -g_MinVx) vx = -g_MinVx;
         else if (vx < -g_MaxVx) vx = -g_MaxVx;
      }

      // Ponemos la pelota justo delante de la paleta (ha chocado)
      x = panx = palX + ((int)signoxPelota);

      // Hay contacto xD
      contacto = true;
   }

   return contacto;
}

//////////////////////////////////////////////////////////////////
// VelX:                                                        //
// * Obtiene la velocidad horizontal de la pelota.              //
//////////////////////////////////////////////////////////////////
float
TPelota::VelX() const { return vx; }

//////////////////////////////////////////////////////////////////
// VelY:                                                        //
// * Obtiene la velocidad vertical de la pelota.                //
//////////////////////////////////////////////////////////////////
float
TPelota::VelY() const { return vy; }

//////////////////////////////////////////////////////////////////
// PosX:                                                        //
// * Obtiene la posicion horizontal REAL de la pelota           //
//////////////////////////////////////////////////////////////////
float
TPelota::PosX() const { return x; }

//////////////////////////////////////////////////////////////////
// PosY:                                                        //
// * Obtiene la posicion vertical REAL de la pelota             //
//////////////////////////////////////////////////////////////////
float
TPelota::PosY() const { return y; }

//////////////////////////////////////////////////////////////////
// Caracter:                                                    //
// * Obtiene el caracter de la pelota.                          //
//////////////////////////////////////////////////////////////////
chtype
TPelota::Caracter() const { return ch; }

//////////////////////////////////////////////////////////////////
// Atributo:                                                    //
// * Obtiene el atributo de la pelota                           //
//////////////////////////////////////////////////////////////////
int
TPelota::Atributo() const { return Atr; }

//////////////////////////////////////////////////////////////////
// ColorTexto:                                                  //
// * Obtiene el Color Texto de la pelota.                       //
//////////////////////////////////////////////////////////////////
int
TPelota::ColorTexto() const { return CT; }

//////////////////////////////////////////////////////////////////
// ColorFondo:                                                  //
// * Obtiene el Color Fondo de la pelota                        //
//////////////////////////////////////////////////////////////////
int
TPelota::ColorFondo() const { return CF; }
