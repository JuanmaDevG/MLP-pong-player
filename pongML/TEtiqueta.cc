#include "TEtiqueta.h"

///////////////////////////////////////////////////////////
// Constructor:	                                         //
// * Inicializamos toas las cosas, pa varial.            //
///////////////////////////////////////////////////////////
TEtiqueta::TEtiqueta (const char *str)
{
   // Definimos ventana y posicion
   x = 0; y = 0;
   minX = 0; minY = 0;
   maxX = STDP::TamX() - 1; maxY = STDP::TamY() - 1;

   // Definimos cadena, tamanyo y atributos
   cadena = NULL; PonCadena ("-");
   Atr = STDP_A_NORMAL; CT = STDP_C_BLANCO; CF = STDP_C_NEGRO;
}

///////////////////////////////////////////////////////////
// Destructor:                                           //
// * Uys! aqui si que hay cosas que destruir :P.         //
///////////////////////////////////////////////////////////
TEtiqueta::~TEtiqueta ()
{
   if (cadena != NULL) { delete [] cadena; cadena = NULL; }
   tamanyo = 0;
}

///////////////////////////////////////////////////////////
// PonCadena:                                            //
// * Con esto ponemos una nueva cadena ;P.               //
///////////////////////////////////////////////////////////
void
TEtiqueta::PonCadena (const char *str)
{
   // Si ya teniamos una cadena con algo, primero la borramos
   if (cadena != NULL) { delete [] cadena; cadena = NULL; }

   // Y ahora ya podemos meter la nueva tranquilamente
   if (str != NULL)
   {
      tamanyo = strlen(str);
      cadena = new char[ tamanyo + 1 ];
      if (cadena == NULL)
      {
         cerr << "No se pudo reservar memoria en TEtiqueta.\n";
         exit (-1);
      }
      strcpy(cadena, str);
   }
   else
   {
      cadena = NULL;
      tamanyo = 0;
   }
}
///////////////////////////////////////////////////////////
// Posicionar:                                           //
// * Poz, pa poner la etiqueta ande nos de la gana xD.   //
///////////////////////////////////////////////////////////
void
TEtiqueta::Posicionar (int px, int py)
{
   // �Ojito O_o! la posicion de la etiqueta es relativa a la ventana
   if (px >= 0 && px <= (maxX-minX) && py >= 0 && py <= (maxY-minY))
   { x=px; y=py; }
}

///////////////////////////////////////////////////////////
// DefineVentana:                                        //
// * Asin encerramos la etiqueta pa que no se escape xD. //
///////////////////////////////////////////////////////////
void
TEtiqueta::DefineVentana (int pminx, int pminy, int pmaxx, int pmaxy)
{
   if (pminx < pmaxx && pminy <= pmaxy && pminx >=0 && pminy >=0 &&
       pmaxx < STDP::TamX() && pmaxy < STDP::TamY() )
   {
      // Mantenemos posiciones relativas
      if (maxX - minX > 0 && maxY - minY > 0)
      {
         x = (pmaxx - pminx) * x / (maxX - minX);
         y = (pmaxy - pminy) * y / (maxY - minY);
      }
      minX = pminx; minY = pminy; maxX = pmaxx; maxY = pmaxy;
   }
}

///////////////////////////////////////////////////////////
// PonAtributos:                                         //
// * Los atributos de toa la vida (sin caracter xD).     //
///////////////////////////////////////////////////////////
void
TEtiqueta::PonAtributos (int pAtr, int pCT, int pCF)
{
   if (STDP::EsAtributo(pAtr)) Atr = pAtr;
   if    (STDP::EsColor (pCT)) CT = pCT;
   if    (STDP::EsColor (pCF)) CF = pCF;
}

///////////////////////////////////////////////////////////
// Dibujar:                                              //
// * Toda! de arriba a abajo, Toda! Entera y tuya...     //
///////////////////////////////////////////////////////////
void
TEtiqueta::Dibujar ()
{
   // Calculamos el espacio que tenemos para dibujar
   int dist = maxX - (minX + x) + 1;

   // Y pintamos la parte que coge de la cadena
   STDP::PonCursor   (minX + x, minY + y);   // �Ojo! �Posicion relativa!
   STDP::CambiaColor (Atr, CT, CF);
   if (dist < tamanyo)
   {
      char cad[dist + 1];
      for (int i=0; i<dist; i++) cad[i] = cadena[i];
      STDP::EscribeStr (cad);
   }
   else
      STDP::EscribeStr (cadena);
}

///////////////////////////////////////////////////////////
// Cadena:                                               //
// * Obtenemos lo que pone en la etiqueta.               //
///////////////////////////////////////////////////////////
char *
TEtiqueta::Cadena () const { return cadena; }

///////////////////////////////////////////////////////////
// Tamanyo:                                              //
// * Desimos cuanto ocupa la cadena.                     //
///////////////////////////////////////////////////////////
int
TEtiqueta::Tamanyo () const { return tamanyo; }

///////////////////////////////////////////////////////////
// PosX:                                                 //
// * La posisiohn horisontaAaal! xD.                     //
///////////////////////////////////////////////////////////
int
TEtiqueta::PosX () const { return x; }

///////////////////////////////////////////////////////////
// PosY:                                                 //
// * La posision vertical.                               //
///////////////////////////////////////////////////////////
int
TEtiqueta::PosY () const { return y; }

///////////////////////////////////////////////////////////
// Atributo:                                             //
// * Laalala, lalalalala, Laalala, lalalalala...         //
///////////////////////////////////////////////////////////
int
TEtiqueta::Atributo () const { return Atr; }

///////////////////////////////////////////////////////////
// ColorTexto:                                           //
// * pray for ever, and ever, and ever...                //
///////////////////////////////////////////////////////////
int
TEtiqueta::ColorTexto () const { return CT; }

///////////////////////////////////////////////////////////
// ColorFondo:                                           //
// * Set me freeeee...pray for ever, and ever, and ever..//
///////////////////////////////////////////////////////////
int
TEtiqueta::ColorFondo () const { return CF; }
