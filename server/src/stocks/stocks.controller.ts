import { Controller, Get, Param, Query } from '@nestjs/common';

@Controller('api/stocks')
export class StocksController {
  @Get()
  getAll(): string{
    return 'This will return all stocks';
  }

  @Get('search')
  search(@Query('name') searchingName: string): string{
    return `This will return stocks whose name contains "${searchingName}"`;
  }

  @Get(':id')
  getOne(@Param('id') id: number): string{
    return `This will return stock ID ${id}`;
  }
}